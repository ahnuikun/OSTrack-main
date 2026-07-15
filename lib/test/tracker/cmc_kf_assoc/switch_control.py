"""Causal margin and camera-consistent hysteresis for candidate reranking."""

from dataclasses import asdict, dataclass

import numpy as np

from .association import box_iou_xywh
from .state_propagation import project_box


@dataclass(frozen=True)
class SwitchControlConfig:
    minimum_final_margin: float = 0.02
    confirmation_frames: int = 2
    minimum_consistency_iou: float = 0.30
    require_valid_cmc: bool = True


@dataclass(frozen=True)
class SwitchControlDecision:
    proposed_index: int
    proposed_rank: int
    selected_index: int
    selected_rank: int
    accepted_rerank: bool
    final_margin: float
    consistency_iou: float | None
    pending_count: int
    reason: str

    def to_dict(self):
        return asdict(self)


class CameraConsistentSwitchController:
    """Commit a non-Top-1 candidate only after a stable causal confirmation."""

    def __init__(self, config=None):
        self.config = config or SwitchControlConfig()
        if self.config.confirmation_frames < 1:
            raise ValueError("confirmation_frames must be positive")
        if not 0.0 <= self.config.minimum_consistency_iou <= 1.0:
            raise ValueError("minimum_consistency_iou must be in [0, 1]")
        self.pending_box = None
        self.pending_count = 0

    def reset(self):
        self.pending_box = None
        self.pending_count = 0

    def _decision(self, proposal, candidates, selected_index, accepted,
                  margin, consistency, reason):
        return SwitchControlDecision(
            proposed_index=int(proposal.selected_index),
            proposed_rank=int(proposal.selected_rank),
            selected_index=int(selected_index),
            selected_rank=int(candidates[selected_index].rank),
            accepted_rerank=bool(accepted), final_margin=float(margin),
            consistency_iou=(None if consistency is None else float(consistency)),
            pending_count=int(self.pending_count), reason=reason)

    def decide(self, candidates, proposal, cmc_result):
        if not candidates:
            raise ValueError("candidate list is empty")
        proposed_index = int(proposal.selected_index)
        if proposed_index < 0 or proposed_index >= len(candidates):
            raise ValueError("proposal selected_index is out of range")
        if len(proposal.final_scores) != len(candidates):
            raise ValueError("proposal score count does not match candidates")

        if proposed_index == 0:
            self.reset()
            return self._decision(
                proposal, candidates, 0, False, 0.0, None, "appearance_top1")

        final_scores = np.asarray(proposal.final_scores, dtype=np.float64)
        margin = float(final_scores[proposed_index] - final_scores[0])
        if not np.isfinite(margin) or margin < self.config.minimum_final_margin:
            self.reset()
            return self._decision(
                proposal, candidates, 0, False, margin, None, "margin_hold")

        if (self.config.require_valid_cmc and
                (cmc_result is None or not bool(cmc_result.valid))):
            self.reset()
            return self._decision(
                proposal, candidates, 0, False, margin, None, "invalid_cmc_hold")

        proposed_box = [float(value) for value in candidates[proposed_index].image_box]
        if self.config.confirmation_frames == 1:
            self.reset()
            return self._decision(
                proposal, candidates, proposed_index, True, margin, None,
                "margin_confirmed_rerank")

        if self.pending_box is None:
            self.pending_box = proposed_box
            self.pending_count = 1
            return self._decision(
                proposal, candidates, 0, False, margin, None, "pending_confirmation")

        try:
            homography = (
                cmc_result.homography if cmc_result is not None
                else np.eye(3, dtype=np.float64))
            projected_pending = project_box(self.pending_box, homography)
            consistency = box_iou_xywh(projected_pending, proposed_box)
        except (ValueError, np.linalg.LinAlgError):
            self.pending_box = proposed_box
            self.pending_count = 1
            return self._decision(
                proposal, candidates, 0, False, margin, None,
                "projection_failed_hold")

        if consistency < self.config.minimum_consistency_iou:
            self.pending_box = proposed_box
            self.pending_count = 1
            return self._decision(
                proposal, candidates, 0, False, margin, consistency,
                "inconsistent_hold")

        self.pending_box = proposed_box
        self.pending_count += 1
        if self.pending_count < self.config.confirmation_frames:
            return self._decision(
                proposal, candidates, 0, False, margin, consistency,
                "pending_confirmation")

        confirmed_count = self.pending_count
        self.reset()
        decision = self._decision(
            proposal, candidates, proposed_index, True, margin, consistency,
            "camera_consistent_rerank")
        return SwitchControlDecision(
            **{**decision.to_dict(), "pending_count": confirmed_count})
