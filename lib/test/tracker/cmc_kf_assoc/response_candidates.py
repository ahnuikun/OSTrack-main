"""Decode multiple boxes from one OSTrack center-head response."""

from dataclasses import asdict, dataclass

import torch


@dataclass
class ResponseCandidate:
    rank: int
    flat_index: int
    grid_x: int
    grid_y: int
    score: float
    normalized_box: list
    image_box: list = None
    raw_score: float = None
    hann_score: float = None

    def to_dict(self):
        return asdict(self)


def _decode_indices(indices, size_map, offset_map, feature_size):
    gather_index = indices.view(1, 1, -1).expand(1, 2, -1)
    sizes = size_map.flatten(2).gather(2, gather_index).squeeze(0).T
    offsets = offset_map.flatten(2).gather(2, gather_index).squeeze(0).T
    grid_y = torch.div(indices, feature_size, rounding_mode="floor")
    grid_x = indices % feature_size
    centers_x = (grid_x.float() + offsets[:, 0]) / feature_size
    centers_y = (grid_y.float() + offsets[:, 1]) / feature_size
    return torch.stack([centers_x, centers_y, sizes[:, 0], sizes[:, 1]], dim=1)


def extract_candidates(score_map, size_map, offset_map, top_k=5,
                       suppression_radius=1):
    """Return spatially distinct peaks; rank 1 is the exact global maximum."""

    if score_map.ndim != 4 or score_map.shape[0] != 1 or score_map.shape[1] != 1:
        raise ValueError(f"expected score map shape [1,1,H,W], got {tuple(score_map.shape)}")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    response = score_map.detach().clone()
    _, _, height, width = response.shape
    if height != width:
        raise ValueError("OSTrack center response must be square")

    indices = []
    scores = []
    working = response[0, 0]
    for _ in range(min(top_k, height * width)):
        flat_index = int(torch.argmax(working).item())
        score = float(working.flatten()[flat_index].item())
        if not torch.isfinite(working.flatten()[flat_index]):
            break
        y, x = divmod(flat_index, width)
        indices.append(flat_index)
        scores.append(score)
        y1, y2 = max(0, y - suppression_radius), min(height, y + suppression_radius + 1)
        x1, x2 = max(0, x - suppression_radius), min(width, x + suppression_radius + 1)
        working[y1:y2, x1:x2] = -torch.inf

    index_tensor = torch.tensor(indices, dtype=torch.long, device=score_map.device)
    boxes = _decode_indices(index_tensor, size_map, offset_map, width)
    candidates = []
    for rank, (flat_index, score, box) in enumerate(zip(indices, scores, boxes), start=1):
        y, x = divmod(flat_index, width)
        candidates.append(ResponseCandidate(
            rank=rank, flat_index=flat_index, grid_x=x, grid_y=y, score=score,
            normalized_box=[float(value) for value in box.detach().cpu().tolist()]))
    return candidates


def _box_iou_normalized(first, second):
    acx, acy, aw, ah = first
    bcx, bcy, bw, bh = second
    ax1, ay1, ax2, ay2 = acx - aw / 2, acy - ah / 2, acx + aw / 2, acy + ah / 2
    bx1, by1, bx2, by2 = bcx - bw / 2, bcy - bh / 2, bcx + bw / 2, bcy + bh / 2
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = max(aw, 0.0) * max(ah, 0.0) + max(bw, 0.0) * max(bh, 0.0) - intersection
    return 0.0 if union <= 0 else intersection / union


def extract_mbpp_candidates(hann_response, raw_response, size_map, offset_map,
                            top_k=5, proposal_count=30, nms_iou=0.8):
    """MBPP-compatible proposal set with official Hann Top-1 kept at rank 1.

    UOSTrack obtains alternatives from the unwindowed response and applies box
    NMS.  We retain the official Hann maximum explicitly so ``lambda=0`` and
    every fallback remain exactly aligned with the OSTrack output.
    """

    official = extract_candidates(
        hann_response, size_map, offset_map, top_k=1, suppression_radius=0)[0]
    flat_raw = raw_response.detach().flatten()
    count = min(int(proposal_count), int(flat_raw.numel()))
    _, indices = torch.topk(flat_raw, count, largest=True, sorted=True)
    boxes = _decode_indices(indices, size_map, offset_map, raw_response.shape[-1])

    proposals = []
    for flat_index, box in zip(indices.tolist(), boxes):
        y, x = divmod(int(flat_index), raw_response.shape[-1])
        normalized_box = [float(value) for value in box.detach().cpu().tolist()]
        if any(_box_iou_normalized(normalized_box, item[2]) > nms_iou for item in proposals):
            continue
        proposals.append((int(flat_index), float(flat_raw[flat_index].item()), normalized_box, x, y))

    official.score = float(flat_raw[official.flat_index].item())
    official.raw_score = official.score
    official.hann_score = float(hann_response.detach().flatten()[official.flat_index].item())
    candidates = [official]
    for flat_index, score, box, x, y in proposals:
        if flat_index == official.flat_index:
            continue
        if len(candidates) >= top_k:
            break
        raw_score = float(flat_raw[flat_index].item())
        hann_score = float(hann_response.detach().flatten()[flat_index].item())
        candidates.append(ResponseCandidate(
            rank=len(candidates) + 1, flat_index=flat_index, grid_x=x,
            grid_y=y, score=score, normalized_box=box,
            raw_score=raw_score, hann_score=hann_score))
    return candidates
