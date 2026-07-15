import numpy as np

from lib.test.tracker.cmc_kf_assoc.association import AssociationDecision
from lib.test.tracker.cmc_kf_assoc.camera_motion import CameraMotionResult
from lib.test.tracker.cmc_kf_assoc.response_candidates import ResponseCandidate
from lib.test.tracker.cmc_kf_assoc.switch_control import (
    CameraConsistentSwitchController,
    SwitchControlConfig,
)


def _candidate(rank, box):
    return ResponseCandidate(
        rank=rank, flat_index=rank - 1, grid_x=rank - 1, grid_y=0,
        score=1.0 / rank, normalized_box=[0, 0, 1, 1], image_box=box)


def _proposal(final_scores=(0.90, 0.91)):
    return AssociationDecision(
        selected_index=1, selected_rank=2, reason="proposal",
        motion_weight=0.25, appearance_scores=[1.0, 0.9],
        motion_scores=[0.5, 0.8], final_scores=list(final_scores),
        q_cmc=1.0, q_kf=1.0, q_ambiguity=1.0)


def _cmc(homography=None, valid=True):
    return CameraMotionResult(
        homography=(np.eye(3) if homography is None else homography),
        valid=valid, quality=1.0 if valid else 0.0, fallback_reason="",
        detected_points=20, tracked_points=20, inliers=20,
        inlier_ratio=1.0, reprojection_error=0.0, coverage=0.5,
        elapsed_ms=1.0)


def test_small_final_margin_holds_appearance_top1():
    controller = CameraConsistentSwitchController(
        SwitchControlConfig(minimum_final_margin=0.02))
    candidates = [_candidate(1, [0, 0, 10, 10]), _candidate(2, [20, 0, 10, 10])]
    decision = controller.decide(candidates, _proposal((0.90, 0.9061)), _cmc())
    assert decision.selected_rank == 1
    assert decision.reason == "margin_hold"
    assert controller.pending_count == 0


def test_two_camera_consistent_proposals_confirm_rerank():
    controller = CameraConsistentSwitchController(SwitchControlConfig(
        minimum_final_margin=0.02, confirmation_frames=2,
        minimum_consistency_iou=0.5))
    candidates_t1 = [_candidate(1, [0, 0, 10, 10]), _candidate(2, [20, 0, 10, 10])]
    first = controller.decide(candidates_t1, _proposal((0.80, 0.90)), _cmc())
    assert first.selected_rank == 1
    assert first.reason == "pending_confirmation"

    translation = np.array([[1, 0, 3], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    candidates_t2 = [_candidate(1, [1, 0, 10, 10]), _candidate(2, [23, 0, 10, 10])]
    second = controller.decide(
        candidates_t2, _proposal((0.80, 0.90)), _cmc(translation))
    assert second.selected_rank == 2
    assert second.accepted_rerank
    assert second.reason == "camera_consistent_rerank"
    assert second.pending_count == 2


def test_inconsistent_or_invalid_cmc_never_accumulates_confirmation():
    controller = CameraConsistentSwitchController(SwitchControlConfig(
        minimum_final_margin=0.02, confirmation_frames=2,
        minimum_consistency_iou=0.5))
    candidates = [_candidate(1, [0, 0, 10, 10]), _candidate(2, [20, 0, 10, 10])]
    controller.decide(candidates, _proposal((0.80, 0.90)), _cmc())
    moved = [_candidate(1, [0, 0, 10, 10]), _candidate(2, [80, 0, 10, 10])]
    inconsistent = controller.decide(moved, _proposal((0.80, 0.90)), _cmc())
    assert inconsistent.selected_rank == 1
    assert inconsistent.reason == "inconsistent_hold"
    assert inconsistent.pending_count == 1

    invalid = controller.decide(moved, _proposal((0.80, 0.90)), _cmc(valid=False))
    assert invalid.selected_rank == 1
    assert invalid.reason == "invalid_cmc_hold"
    assert controller.pending_count == 0
