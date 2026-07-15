"""Pure observation-write policy kept separate from output selection."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservationDecision:
    accepted: bool
    reason: str


def decide_observation(selected_rank, appearance_score, q_ambiguity, mahalanobis,
                       policy, minimum_score=0.15, strong_score=0.25,
                       chi2_threshold=13.2767):
    score_ok = float(appearance_score) >= float(minimum_score)
    innovation_ok = float(mahalanobis) <= float(chi2_threshold)
    if policy == "clear_top1_or_consistent":
        clear_top1 = int(selected_rank) == 1 and float(q_ambiguity) <= 0.5
        accepted = bool(clear_top1 or (score_ok and innovation_ok))
        if accepted:
            reason = "clear_top1" if clear_top1 else "consistent_measurement"
        else:
            reason = (
                "ambiguous_low_response" if not score_ok
                else "ambiguous_large_innovation")
        return ObservationDecision(accepted, reason)
    if policy == "legacy_absolute":
        strong_top1 = int(selected_rank) == 1 and float(appearance_score) >= float(strong_score)
        accepted = bool(score_ok and (innovation_ok or strong_top1))
        if accepted:
            reason = "strong_top1" if strong_top1 else "consistent_measurement"
        else:
            reason = "low_response" if not score_ok else "large_innovation"
        return ObservationDecision(accepted, reason)
    raise ValueError(f"unknown observation policy: {policy}")
