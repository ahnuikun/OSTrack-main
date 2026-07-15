from lib.test.tracker.cmc_kf_assoc.observation import decide_observation


def test_clear_top1_can_recover_even_when_absolute_score_is_low():
    decision = decide_observation(
        selected_rank=1, appearance_score=0.05, q_ambiguity=0.2,
        mahalanobis=50.0, policy="clear_top1_or_consistent")
    assert decision.accepted
    assert decision.reason == "clear_top1"


def test_ambiguous_inconsistent_measurement_is_rejected():
    decision = decide_observation(
        selected_rank=2, appearance_score=0.4, q_ambiguity=0.9,
        mahalanobis=50.0, policy="clear_top1_or_consistent")
    assert not decision.accepted
    assert decision.reason == "ambiguous_large_innovation"


def test_consistent_alternative_can_update():
    decision = decide_observation(
        selected_rank=2, appearance_score=0.4, q_ambiguity=0.9,
        mahalanobis=2.0, policy="clear_top1_or_consistent")
    assert decision.accepted
    assert decision.reason == "consistent_measurement"
