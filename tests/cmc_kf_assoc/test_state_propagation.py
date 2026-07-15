import numpy as np

from lib.test.tracker.cmc_kf_assoc.state_propagation import (
    propagate_state_and_covariance,
    propagate_state_vector,
)


def test_identity_preserves_state_and_covariance():
    state = np.array([100.0, 80.0, 30.0, 20.0, 4.0, -2.0, 1.0, 0.5])
    covariance = np.diag(np.arange(1.0, 9.0))
    propagated, propagated_covariance, jacobian = propagate_state_and_covariance(
        state, covariance, np.eye(3))
    np.testing.assert_allclose(propagated, state, atol=1e-8)
    np.testing.assert_allclose(jacobian, np.eye(8), atol=1e-6)
    np.testing.assert_allclose(propagated_covariance, covariance, atol=1e-5)


def test_translation_moves_center_without_changing_target_velocity():
    state = np.array([100.0, 80.0, 30.0, 20.0, 4.0, -2.0, 1.0, 0.5])
    homography = np.array([[1.0, 0.0, 12.0], [0.0, 1.0, -7.0], [0.0, 0.0, 1.0]])
    propagated = propagate_state_vector(state, homography)
    expected = state.copy()
    expected[0] += 12.0
    expected[1] -= 7.0
    np.testing.assert_allclose(propagated, expected, atol=1e-8)
