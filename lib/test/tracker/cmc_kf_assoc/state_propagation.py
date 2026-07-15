"""Geometry for propagating a box Kalman state through camera motion."""

import numpy as np


EPS = 1e-8


def box_to_measurement(box_xywh):
    x, y, w, h = np.asarray(box_xywh, dtype=np.float64)
    return np.array([x + 0.5 * w, y + 0.5 * h, w, h], dtype=np.float64)


def measurement_to_box(measurement):
    cx, cy, w, h = np.asarray(measurement, dtype=np.float64)
    w = max(float(w), EPS)
    h = max(float(h), EPS)
    return np.array([cx - 0.5 * w, cy - 0.5 * h, w, h], dtype=np.float64)


def project_points(points, homography):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homography = np.asarray(homography, dtype=np.float64).reshape(3, 3)
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    projected = homogeneous @ homography.T
    denominator = projected[:, 2]
    if np.any(np.abs(denominator) < EPS):
        raise ValueError("homography projects a point to infinity")
    result = projected[:, :2] / denominator[:, None]
    if not np.all(np.isfinite(result)):
        raise ValueError("homography produced non-finite coordinates")
    return result


def project_box(box_xywh, homography):
    x, y, w, h = np.asarray(box_xywh, dtype=np.float64)
    corners = np.array(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        dtype=np.float64)
    warped = project_points(corners, homography)
    minimum = warped.min(axis=0)
    maximum = warped.max(axis=0)
    size = np.maximum(maximum - minimum, EPS)
    return np.array([minimum[0], minimum[1], size[0], size[1]], dtype=np.float64)


def propagate_state_vector(state, homography):
    """Warp current box and its one-step target-motion endpoint consistently."""

    state = np.asarray(state, dtype=np.float64).reshape(8)
    current_box = measurement_to_box(state[:4])
    future_measurement = state[:4] + state[4:]
    future_measurement[2:] = np.maximum(future_measurement[2:], EPS)
    future_box = measurement_to_box(future_measurement)

    current_warped = box_to_measurement(project_box(current_box, homography))
    future_warped = box_to_measurement(project_box(future_box, homography))
    result = np.concatenate([current_warped, future_warped - current_warped])
    if not np.all(np.isfinite(result)):
        raise ValueError("camera propagation produced a non-finite state")
    result[2:4] = np.maximum(result[2:4], EPS)
    return result


def numerical_state_jacobian(state, homography):
    state = np.asarray(state, dtype=np.float64).reshape(8)
    jacobian = np.zeros((8, 8), dtype=np.float64)
    for index in range(8):
        step = max(abs(float(state[index])) * 1e-5, 1e-4)
        plus = state.copy()
        minus = state.copy()
        plus[index] += step
        minus[index] -= step
        if index in (2, 3) and minus[index] <= EPS:
            base = propagate_state_vector(state, homography)
            forward = propagate_state_vector(plus, homography)
            jacobian[:, index] = (forward - base) / step
        else:
            forward = propagate_state_vector(plus, homography)
            backward = propagate_state_vector(minus, homography)
            jacobian[:, index] = (forward - backward) / (2.0 * step)
    return jacobian


def propagate_state_and_covariance(state, covariance, homography, noise=None):
    state = np.asarray(state, dtype=np.float64).reshape(8)
    covariance = np.asarray(covariance, dtype=np.float64).reshape(8, 8)
    propagated = propagate_state_vector(state, homography)
    jacobian = numerical_state_jacobian(state, homography)
    result_covariance = jacobian @ covariance @ jacobian.T
    if noise is not None:
        result_covariance = result_covariance + np.asarray(noise, dtype=np.float64)
    result_covariance = 0.5 * (result_covariance + result_covariance.T)
    if not np.all(np.isfinite(result_covariance)):
        raise ValueError("camera propagation produced a non-finite covariance")
    return propagated, result_covariance, jacobian
