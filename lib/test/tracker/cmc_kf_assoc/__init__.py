"""Inference-only CMC, Kalman prediction, and response association."""

from .association import AssociationConfig, AssociationDecision, associate_candidates
from .camera_motion import CameraMotionConfig, CameraMotionEstimator, CameraMotionResult
from .kalman_box import KalmanBoxFilter
from .response_candidates import ResponseCandidate, extract_candidates
from .switch_control import (
    CameraConsistentSwitchController,
    SwitchControlConfig,
    SwitchControlDecision,
)

__all__ = [
    "AssociationConfig",
    "AssociationDecision",
    "CameraMotionConfig",
    "CameraMotionEstimator",
    "CameraMotionResult",
    "KalmanBoxFilter",
    "ResponseCandidate",
    "CameraConsistentSwitchController",
    "SwitchControlConfig",
    "SwitchControlDecision",
    "associate_candidates",
    "extract_candidates",
]
