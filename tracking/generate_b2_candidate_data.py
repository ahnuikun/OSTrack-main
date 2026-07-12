"""Generate training-only B1 Top-K candidate association pairs from GOT-10k."""

import argparse
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from lib.models.ostrack.candidate_association import CandidateAssociationNet
from lib.test.parameter.ostrack import parameters
from lib.test.tracker.ostrack import OSTrack
from lib.train.data import opencv_loader
from lib.train.dataset.got10k import Got10k


def xywh_iou(boxes, gt):
    """IoU between K xywh boxes and one xywh ground-truth box."""
    boxes = boxes.float()
    gt = gt.float()
    boxes_xy2 = boxes[:, :2] + boxes[:, 2:].clamp_min(0)
    gt_xy2 = gt[:2] + gt[2:].clamp_min(0)
    inter_lt = torch.maximum(boxes[:, :2], gt[:2])
    inter_rb = torch.minimum(boxes_xy2, gt_xy2)
    inter = (inter_rb - inter_lt).clamp_min(0).prod(dim=1)
    union = boxes[:, 2:].clamp_min(0).prod(dim=1) + gt[2:].clamp_min(0).prod() - inter
    return inter / union.clamp_min(1e-6)


def parse_args():
    parser = argparse.ArgumentParser(description='Collect B1 free-run Top-K candidate pairs on GOT-10k train.')
    parser.add_argument('--config', default='vitb_256_mae_ce_32x4_ep300_fulltn_b2_collect')
    parser.add_argument('--checkpoint', required=True, help='Frozen 300-epoch OSTrack checkpoint.')
    parser.add_argument('--output', default='output/b2_candidate_assoc/got10k_b1_pairs.pt')
    parser.add_argument('--max_sequences', type=int, default=120)
    parser.add_argument('--max_frames', type=int, default=300)
    parser.add_argument('--positive_iou', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=20260711)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Build a deterministic, broadly distributed subset without touching any test split.
    list_path = os.path.join(PROJECT_ROOT, 'data', 'got10k', 'train', 'list.txt')
    with open(list_path, 'r') as handle:
        total_sequences = len([line for line in handle if line.strip()])
    selected_ids = np.linspace(0, total_sequences - 1, min(args.max_sequences, total_sequences), dtype=np.int64).tolist()
    # The local workspace lacks jpeg4py's native dependency; use the same RGB
    # OpenCV fallback used by the test-time preprocessor.
    dataset = Got10k(seq_ids=selected_ids, image_loader=opencv_loader)

    params = parameters(args.config)
    params.checkpoint = os.path.abspath(args.checkpoint)
    params.debug = 0
    params.save_all_boxes = False
    tracker = OSTrack(params, 'got10k_train')
    compressor = CandidateAssociationNet(
        compressed_dim=int(params.cfg.TEST.CANDIDATE_ASSOC.COMPRESSED_DIM),
        hidden_dim=int(params.cfg.TEST.CANDIDATE_ASSOC.HIDDEN_DIM),
        embedding_dim=int(params.cfg.TEST.CANDIDATE_ASSOC.EMBEDDING_DIM),
        temperature=float(params.cfg.TEST.CANDIDATE_ASSOC.TEMPERATURE)).cuda().eval()

    train_samples, val_samples = [], []
    total_frames = recalled_frames = positive_pairs = dustbin_pairs = 0
    sequence_stats = []
    with torch.no_grad():
        for local_id in range(dataset.get_num_sequences()):
            sequence_name = dataset.sequence_list[local_id]
            sequence_info = dataset.get_sequence_info(local_id)
            frame_count = min(len(sequence_info['bbox']), args.max_frames)
            if frame_count < 3:
                continue
            initial_image = dataset.get_frames(local_id, [0], anno=sequence_info)[0][0]
            tracker.initialize(initial_image, {'init_bbox': sequence_info['bbox'][0].tolist()})
            previous_target = None
            sequence_pairs = sequence_recalled = 0
            is_validation = (local_id % 5 == 0)

            for frame_id in range(1, frame_count):
                image = dataset.get_frames(local_id, [frame_id], anno=sequence_info)[0][0]
                tracker.track(image)
                record = tracker.last_candidate_record
                if record is None:
                    raise RuntimeError('B2 collector did not receive a Top-K candidate record.')
                ious = xywh_iou(record['boxes_image'].detach().cpu(), sequence_info['bbox'][frame_id].cpu())
                best_iou, target_index = torch.max(ious, dim=0)
                label = int(target_index.item()) if float(best_iou) >= args.positive_iou else int(ious.numel())
                total_frames += 1
                if label < ious.numel():
                    recalled_frames += 1
                    sequence_recalled += 1

                deep, shallow = compressor.compress(record['deep'], record['shallow'])
                current = {
                    'deep': deep.detach().cpu().half(),
                    'shallow': shallow.detach().cpu().half(),
                    'geometry': record['geometry'].detach().cpu().half(),
                }
                if previous_target is not None:
                    sample = {
                        'prev_deep': previous_target['deep'],
                        'prev_shallow': previous_target['shallow'],
                        'prev_geometry': previous_target['geometry'],
                        'curr_deep': current['deep'],
                        'curr_shallow': current['shallow'],
                        'curr_geometry': current['geometry'],
                        'label': label,
                    }
                    (val_samples if is_validation else train_samples).append(sample)
                    sequence_pairs += 1
                    if label < ious.numel():
                        positive_pairs += 1
                    else:
                        dustbin_pairs += 1

                if label < ious.numel():
                    previous_target = {
                        'deep': current['deep'][label].clone(),
                        'shallow': current['shallow'][label].clone(),
                        'geometry': current['geometry'][label].clone(),
                    }
                else:
                    previous_target = None

            sequence_stats.append({
                'name': sequence_name,
                'frames': frame_count - 1,
                'pairs': sequence_pairs,
                'topk_recall': sequence_recalled / max(frame_count - 1, 1),
                'split': 'val' if is_validation else 'train',
            })
            print('B2 collect {:3d}/{:3d}: {:s}, pairs={}, recall={:.3f}'.format(
                local_id + 1, dataset.get_num_sequences(), sequence_name, sequence_pairs,
                sequence_recalled / max(frame_count - 1, 1)), flush=True)

    payload = {
        'train_samples': train_samples,
        'val_samples': val_samples,
        'sequence_stats': sequence_stats,
        'metadata': {
            'source': 'got10k/train',
            'selected_sequences': len(sequence_stats),
            'max_frames': args.max_frames,
            'positive_iou': args.positive_iou,
            'topk_recall': recalled_frames / max(total_frames, 1),
            'total_frames': total_frames,
            'positive_pairs': positive_pairs,
            'dustbin_pairs': dustbin_pairs,
            'config': args.config,
            'checkpoint': params.checkpoint,
        },
    }
    torch.save(payload, output_path)
    print('Saved {} train and {} validation pairs to {}'.format(
        len(train_samples), len(val_samples), output_path))
    print('Top-K recall={:.4f}, positives={}, dustbins={}'.format(
        payload['metadata']['topk_recall'], positive_pairs, dustbin_pairs))


if __name__ == '__main__':
    main()
