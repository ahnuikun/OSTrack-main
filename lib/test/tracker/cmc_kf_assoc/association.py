"""Published baselines and reliability-controlled candidate association."""

from dataclasses import asdict, dataclass

import numpy as np


def box_iou_xywh(first, second):
    ax, ay, aw, ah = [float(value) for value in first]
    bx, by, bw, bh = [float(value) for value in second]
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(aw, 0.0) * max(ah, 0.0) + max(bw, 0.0) * max(bh, 0.0) - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


@dataclass(frozen=True)
class AssociationConfig:
    backend: str = "appearance_top1"
    fixed_motion_weight: float = 0.25
    maximum_motion_weight: float = 0.35
    mbpp_keep_iou: float = 0.60
    ambiguity_margin: float = 0.15


@dataclass
class AssociationDecision:
    selected_index: int
    selected_rank: int
    reason: str
    motion_weight: float
    appearance_scores: list
    motion_scores: list
    final_scores: list
    q_cmc: float
    q_kf: float
    q_ambiguity: float

    def to_dict(self):
        return asdict(self)


def _normalized_appearance(candidates):
    scores = np.asarray([candidate.score for candidate in candidates], dtype=np.float64)
    if len(scores) == 0:
        raise ValueError("candidate list is empty")
    minimum, maximum = float(scores.min()), float(scores.max())
    if maximum - minimum < 1e-12:
        return np.ones_like(scores)
    return (scores - minimum) / (maximum - minimum)


def ambiguity_quality(candidates, margin_scale=0.15):
    if len(candidates) < 2:
        return 0.0
    top = max(float(candidates[0].score), 1e-6)
    relative_margin = (float(candidates[0].score) - float(candidates[1].score)) / top
    return float(np.clip(1.0 - relative_margin / max(margin_scale, 1e-6), 0.0, 1.0))


def associate_candidates(candidates, predicted_box, config=None,
                         q_cmc=1.0, q_kf=1.0, q_ambiguity=None):
    config = config or AssociationConfig()
    appearance = _normalized_appearance(candidates)
    motion = np.asarray(
        [box_iou_xywh(candidate.image_box, predicted_box) for candidate in candidates],
        dtype=np.float64)
    q_cmc = float(np.clip(q_cmc, 0.0, 1.0))
    q_kf = float(np.clip(q_kf, 0.0, 1.0))
    if q_ambiguity is None:
        q_ambiguity = ambiguity_quality(candidates, config.ambiguity_margin)
    q_ambiguity = float(np.clip(q_ambiguity, 0.0, 1.0))

    if config.backend == "appearance_top1":
        selected, weight, final, reason = 0, 0.0, appearance, "appearance_top1"
    elif config.backend == "mbpp_product":
        final = appearance * motion
        if motion[0] > config.mbpp_keep_iou:
            selected, reason = 0, "mbpp_keep_top1"
        else:
            selected, reason = int(np.argmax(final)), "mbpp_product_rerank"
        weight = 1.0
    elif config.backend == "fixed_iou":
        weight = float(np.clip(config.fixed_motion_weight, 0.0, 1.0))
        final = (1.0 - weight) * appearance + weight * motion
        selected, reason = int(np.argmax(final)), "fixed_iou_rerank"
    elif config.backend == "reliability_iou":
        weight = float(np.clip(
            config.maximum_motion_weight * q_cmc * q_kf * q_ambiguity,
            0.0, 1.0))
        final = (1.0 - weight) * appearance + weight * motion
        selected, reason = int(np.argmax(final)), "reliability_iou_rerank"
    else:
        raise ValueError(f"unknown association backend: {config.backend}")

    return AssociationDecision(
        selected_index=selected, selected_rank=candidates[selected].rank,
        reason=reason, motion_weight=weight,
        appearance_scores=appearance.tolist(), motion_scores=motion.tolist(),
        final_scores=np.asarray(final).tolist(), q_cmc=q_cmc, q_kf=q_kf,
        q_ambiguity=q_ambiguity)
