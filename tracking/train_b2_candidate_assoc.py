"""Train B2's small association head while keeping OSTrack fully frozen."""

import argparse
import os
import random
import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from lib.models.ostrack.candidate_association import CandidateAssociationNet


def parse_args():
    parser = argparse.ArgumentParser(description='Train B2 candidate association from frozen B1 trajectory pairs.')
    parser.add_argument('--data', default='output/b2_candidate_assoc/got10k_b1_pairs.pt')
    parser.add_argument('--output', default='output/b2_candidate_assoc/b2_assoc.pt')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=20260711)
    return parser.parse_args()


def make_batch(samples, indices, device):
    keys = ('prev_deep', 'prev_shallow', 'prev_geometry', 'curr_deep', 'curr_shallow', 'curr_geometry')
    batch = {key: torch.stack([samples[i][key] for i in indices]).to(device=device, dtype=torch.float32)
             for key in keys}
    labels = torch.tensor([samples[i]['label'] for i in indices], device=device, dtype=torch.long)
    return batch, labels


def evaluate(model, samples, batch_size, device):
    if not samples:
        return {'loss': float('nan'), 'accuracy': float('nan'), 'target_accuracy': float('nan'), 'dustbin_accuracy': float('nan')}
    model.eval()
    losses, predictions, labels_all = [], [], []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch, labels = make_batch(samples, range(start, min(start + batch_size, len(samples))), device)
            previous = model.encode_compressed(batch['prev_deep'], batch['prev_shallow'], batch['prev_geometry'])
            current = model.encode_compressed(batch['curr_deep'], batch['curr_shallow'], batch['curr_geometry'])
            logits = model.logits(previous, current, batch['curr_geometry'])
            losses.append(F.cross_entropy(logits, labels).item() * labels.numel())
            predictions.append(logits.argmax(dim=1).cpu())
            labels_all.append(labels.cpu())
    predictions = torch.cat(predictions)
    labels_all = torch.cat(labels_all)
    target_mask = labels_all < (predictions.max().item() if predictions.numel() else 0)
    # The dustbin class is exactly K; K is inferred from the logits, not the observed predictions.
    dustbin_class = samples[0]['curr_deep'].shape[0]
    target_mask = labels_all < dustbin_class
    dustbin_mask = ~target_mask
    return {
        'loss': sum(losses) / len(samples),
        'accuracy': (predictions == labels_all).float().mean().item(),
        'target_accuracy': (predictions[target_mask] == labels_all[target_mask]).float().mean().item()
        if target_mask.any() else float('nan'),
        'dustbin_accuracy': (predictions[dustbin_mask] == labels_all[dustbin_mask]).float().mean().item()
        if dustbin_mask.any() else float('nan'),
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_path = os.path.abspath(args.data)
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = torch.load(data_path, map_location='cpu', weights_only=False)
    train_samples, val_samples = payload['train_samples'], payload['val_samples']
    if not train_samples or not val_samples:
        raise RuntimeError('B2 needs non-empty train and validation candidate pairs.')

    compressed_dim = train_samples[0]['curr_deep'].shape[-1]
    model = CandidateAssociationNet(compressed_dim=compressed_dim).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    device = torch.device('cuda')
    candidate_count = train_samples[0]['curr_deep'].shape[0]
    dustbin_count = sum(sample['label'] == candidate_count for sample in train_samples)
    target_count = len(train_samples) - dustbin_count
    class_weights = torch.ones(candidate_count + 1, device=device)
    # Free-running B1 still recalls the target in most training frames.  A
    # bounded dustbin weight keeps the no-match state learnable without making
    # a few rare misses dominate the association objective.
    if dustbin_count:
        class_weights[-1] = min(12.0, target_count / dustbin_count)
    print('train pairs={}, dustbins={}, dustbin_weight={:.3f}'.format(
        len(train_samples), dustbin_count, class_weights[-1].item()), flush=True)
    best_state, best_metrics = None, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = list(range(len(train_samples)))
        random.shuffle(indices)
        total_loss = 0.0
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start:start + args.batch_size]
            batch, labels = make_batch(train_samples, batch_indices, device)
            previous = model.encode_compressed(batch['prev_deep'], batch['prev_shallow'], batch['prev_geometry'])
            current = model.encode_compressed(batch['curr_deep'], batch['curr_shallow'], batch['curr_geometry'])
            logits = model.logits(previous, current, batch['curr_geometry'])
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_indices)
        metrics = evaluate(model, val_samples, args.batch_size, device)
        print('epoch {:02d}: train_loss={:.4f} val_loss={:.4f} val_acc={:.4f} target_acc={:.4f} dustbin_acc={:.4f}'.format(
            epoch, total_loss / len(train_samples), metrics['loss'], metrics['accuracy'],
            metrics['target_accuracy'], metrics['dustbin_accuracy']), flush=True)
        if best_metrics is None or metrics['accuracy'] > best_metrics['accuracy']:
            best_metrics = metrics
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    torch.save({
        'model_state': best_state,
        'best_validation': best_metrics,
        'data_metadata': payload['metadata'],
        'train_pairs': len(train_samples),
        'val_pairs': len(val_samples),
    }, output_path)
    print('Saved B2 association checkpoint to {}'.format(output_path))
    print('Best validation: {}'.format(best_metrics))


if __name__ == '__main__':
    main()
