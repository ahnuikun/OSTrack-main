import numpy as np

from lib.test.tracker.cmc_kf_assoc.kalman_box import KalmanBoxFilter


def test_predict_update_remains_finite_and_reduces_measurement_residual():
    tracker = KalmanBoxFilter([10.0, 20.0, 30.0, 40.0])
    tracker.predict()
    measurement = [14.0, 21.0, 30.0, 40.0]
    _, before = tracker.innovation(measurement)
    tracker.update(measurement)
    _, after = tracker.innovation(measurement)
    assert after < before
    assert np.isfinite(tracker.mean).all()
    assert np.isfinite(tracker.covariance).all()
    assert np.linalg.eigvalsh(tracker.covariance).min() >= -1e-8


def test_camera_translation_is_applied_before_prediction():
    tracker = KalmanBoxFilter([10.0, 20.0, 30.0, 40.0])
    homography = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]])
    tracker.propagate_camera(homography, quality=1.0)
    predicted = tracker.predict()
    np.testing.assert_allclose(predicted, [15.0, 17.0, 30.0, 40.0], atol=1e-6)
