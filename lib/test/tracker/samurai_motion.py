"""Training-free motion-aware selection used by the SAMURAI test-time ablation.

This is deliberately independent of SAM2 memory attention.  It maintains a
constant-velocity Kalman state for an OSTrack bounding box and ranks OSTrack
response-map candidates by appearance confidence and predicted-box agreement.
"""

import numpy as np


def xywh_to_cxcywh(box):
    x, y, w, h = box
    return np.asarray([x + 0.5 * w, y + 0.5 * h, w, h], dtype=np.float32)


def cxcywh_to_xywh(box):
    cx, cy, w, h = box
    return np.asarray([cx - 0.5 * w, cy - 0.5 * h, w, h], dtype=np.float32)


def box_iou_xywh(box_a, box_b):
    """IoU for two xywh boxes; malformed/non-overlapping boxes return zero."""
    ax, ay, aw, ah = np.asarray(box_a, dtype=np.float32)
    bx, by, bw, bh = np.asarray(box_b, dtype=np.float32)
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    inter_w = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    inter_h = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


class BoxKalmanFilter:
    """Eight-state constant-velocity Kalman filter over (cx, cy, w, h)."""

    def __init__(self, initial_box, process_noise=1.0, measurement_noise=10.0):
        self.motion_mat = np.eye(8, dtype=np.float32)
        self.motion_mat[:4, 4:] = np.eye(4, dtype=np.float32)
        self.update_mat = np.zeros((4, 8), dtype=np.float32)
        self.update_mat[:, :4] = np.eye(4, dtype=np.float32)
        self.process_cov = np.eye(8, dtype=np.float32) * float(process_noise)
        self.measurement_cov = np.eye(4, dtype=np.float32) * float(measurement_noise)
        self.mean = np.zeros(8, dtype=np.float32)
        self.mean[:4] = xywh_to_cxcywh(initial_box)
        self.covariance = np.eye(8, dtype=np.float32) * float(measurement_noise)

    def predict(self):
        self.mean = self.motion_mat @ self.mean
        self.covariance = self.motion_mat @ self.covariance @ self.motion_mat.T + self.process_cov
        return cxcywh_to_xywh(self.mean[:4]).tolist()

    def update(self, measurement_box):
        measurement = xywh_to_cxcywh(measurement_box)
        projected_cov = self.update_mat @ self.covariance @ self.update_mat.T + self.measurement_cov
        gain = self.covariance @ self.update_mat.T @ np.linalg.inv(projected_cov)
        innovation = measurement - self.update_mat @ self.mean
        self.mean = self.mean + gain @ innovation
        self.covariance = (np.eye(8, dtype=np.float32) - gain @ self.update_mat) @ self.covariance
