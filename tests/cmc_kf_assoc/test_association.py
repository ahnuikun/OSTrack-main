from lib.test.tracker.cmc_kf_assoc.association import (
    AssociationConfig,
    associate_candidates,
)
from lib.test.tracker.cmc_kf_assoc.response_candidates import ResponseCandidate


def _candidate(rank, score, box):
    return ResponseCandidate(rank, rank - 1, rank - 1, 0, score, [0, 0, 0, 0], box)


def test_appearance_backend_is_output_identical_to_rank_one():
    candidates = [_candidate(1, 0.9, [0, 0, 10, 10]), _candidate(2, 0.8, [20, 0, 10, 10])]
    decision = associate_candidates(
        candidates, [20, 0, 10, 10], AssociationConfig(backend="appearance_top1"))
    assert decision.selected_rank == 1
    assert decision.motion_weight == 0.0


def test_mbpp_keeps_top1_when_prediction_iou_is_high():
    candidates = [_candidate(1, 0.9, [0, 0, 10, 10]), _candidate(2, 0.8, [2, 0, 10, 10])]
    decision = associate_candidates(
        candidates, [0, 0, 10, 10], AssociationConfig(backend="mbpp_product"))
    assert decision.selected_rank == 1
    assert decision.reason == "mbpp_keep_top1"


def test_reliability_zero_cannot_override_top1():
    candidates = [_candidate(1, 0.9, [0, 0, 10, 10]), _candidate(2, 0.89, [20, 0, 10, 10])]
    decision = associate_candidates(
        candidates, [20, 0, 10, 10],
        AssociationConfig(backend="reliability_iou", maximum_motion_weight=0.9),
        q_cmc=0.0, q_kf=1.0, q_ambiguity=1.0)
    assert decision.selected_rank == 1
    assert decision.motion_weight == 0.0
