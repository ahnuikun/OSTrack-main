import cv2
import numpy as np

from lib.test.tracker.cmc_kf_assoc.camera_motion import (
    CameraMotionConfig,
    CameraMotionEstimator,
)


def test_synthetic_translation_is_recovered_from_rgb_frames():
    rng = np.random.default_rng(7)
    gray = rng.integers(0, 256, size=(180, 240), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    previous = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    transform = np.float32([[1, 0, 6], [0, 1, -4]])
    current = cv2.warpAffine(previous, transform, (240, 180))
    estimator = CameraMotionEstimator(CameraMotionConfig(min_coverage=0.02))
    result = estimator.estimate(previous, current, target_box=[90, 60, 40, 30])
    assert result.valid, result.fallback_reason
    np.testing.assert_allclose(result.homography[0, 2], 6.0, atol=0.6)
    np.testing.assert_allclose(result.homography[1, 2], -4.0, atol=0.6)


def test_blank_frames_fall_back_to_identity():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    result = CameraMotionEstimator().estimate(frame, frame, target_box=[10, 10, 20, 20])
    assert not result.valid
    assert result.fallback_reason == "insufficient_features"
    np.testing.assert_allclose(result.homography, np.eye(3))
