"""Numerically defensive constant-velocity Kalman filter for image boxes."""

import numpy as np

from .state_propagation import (
    box_to_measurement,
    measurement_to_box,
    propagate_state_and_covariance,
)


class KalmanBoxFilter:
    """State is ``[cx, cy, w, h, vx, vy, vw, vh]`` in image pixels."""

    def __init__(self, box_xywh, process_position=0.05, process_velocity=0.01,
                 measurement_noise=0.05):
        measurement = box_to_measurement(box_xywh)
        self.mean = np.concatenate([measurement, np.zeros(4, dtype=np.float64)])
        scale = max(float(np.sqrt(measurement[2] * measurement[3])), 1.0)
        std = np.array(
            [0.1 * scale, 0.1 * scale, 0.1 * scale, 0.1 * scale,
             0.5 * scale, 0.5 * scale, 0.25 * scale, 0.25 * scale],
            dtype=np.float64)
        self.covariance = np.diag(std ** 2)
        self.process_position = float(process_position)
        self.process_velocity = float(process_velocity)
        self.measurement_noise = float(measurement_noise)
        self.age = 0
        self.predict_only_count = 0

        self.transition = np.eye(8, dtype=np.float64)
        self.transition[:4, 4:] = np.eye(4, dtype=np.float64)
        self.observation = np.zeros((4, 8), dtype=np.float64)
        self.observation[:, :4] = np.eye(4, dtype=np.float64)

    def _scale(self):
        return max(float(np.sqrt(max(self.mean[2], 1.0) * max(self.mean[3], 1.0))), 1.0)

    def _process_covariance(self):
        scale = self._scale()
        std = np.array(
            [self.process_position * scale] * 4 +
            [self.process_velocity * scale] * 4,
            dtype=np.float64)
        return np.diag(std ** 2)

    def _measurement_covariance(self):
        std = np.full(4, self.measurement_noise * self._scale(), dtype=np.float64)
        return np.diag(std ** 2)

    def propagate_camera(self, homography, quality=1.0):
        quality = float(np.clip(quality, 0.0, 1.0))
        scale = self._scale()
        extra_std = (1.0 - quality) * np.array(
            [0.2 * scale] * 4 + [0.1 * scale] * 4, dtype=np.float64)
        noise = np.diag(extra_std ** 2)
        self.mean, self.covariance, jacobian = propagate_state_and_covariance(
            self.mean, self.covariance, homography, noise=noise)
        return jacobian

    def predict(self):
        self.mean = self.transition @ self.mean
        self.mean[2:4] = np.maximum(self.mean[2:4], 1e-3)
        self.covariance = (
            self.transition @ self.covariance @ self.transition.T +
            self._process_covariance())
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.age += 1
        self.predict_only_count += 1
        return self.box

    def project(self):
        measurement_mean = self.observation @ self.mean
        innovation_covariance = (
            self.observation @ self.covariance @ self.observation.T +
            self._measurement_covariance())
        return measurement_mean, innovation_covariance

    def innovation(self, box_xywh):
        measurement = box_to_measurement(box_xywh)
        mean, covariance = self.project()
        residual = measurement - mean
        solution = np.linalg.solve(covariance, residual)
        mahalanobis = float(residual.T @ solution)
        return residual, mahalanobis

    def update(self, box_xywh):
        measurement = box_to_measurement(box_xywh)
        projected_mean, innovation_covariance = self.project()
        residual = measurement - projected_mean
        gain = np.linalg.solve(
            innovation_covariance.T,
            (self.covariance @ self.observation.T).T).T
        self.mean = self.mean + gain @ residual
        self.mean[2:4] = np.maximum(self.mean[2:4], 1e-3)

        identity = np.eye(8, dtype=np.float64)
        correction = identity - gain @ self.observation
        measurement_covariance = self._measurement_covariance()
        self.covariance = (
            correction @ self.covariance @ correction.T +
            gain @ measurement_covariance @ gain.T)
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.predict_only_count = 0
        return residual

    @property
    def box(self):
        return measurement_to_box(self.mean[:4]).tolist()

    @property
    def position_std(self):
        return float(np.sqrt(max(np.trace(self.covariance[:2, :2]) / 2.0, 0.0)))

    @property
    def quality(self):
        normalized_std = self.position_std / self._scale()
        age_penalty = 1.0 / (1.0 + 0.25 * self.predict_only_count)
        return float(np.clip((1.0 / (1.0 + normalized_std)) * age_penalty, 0.0, 1.0))
