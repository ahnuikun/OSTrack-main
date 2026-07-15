"""Sparse-background optical-flow camera motion estimation."""

from dataclasses import asdict, dataclass
from time import perf_counter

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraMotionConfig:
    max_corners: int = 600
    quality_level: float = 0.01
    min_distance: float = 7.0
    block_size: int = 7
    min_correspondences: int = 12
    ransac_threshold: float = 3.0
    min_inlier_ratio: float = 0.45
    max_reprojection_error: float = 3.0
    min_coverage: float = 0.04
    exclusion_margin: float = 0.15
    max_forward_backward_error: float = 1.5
    min_frame_area_ratio: float = 0.25
    max_frame_area_ratio: float = 4.0


@dataclass
class CameraMotionResult:
    homography: np.ndarray
    valid: bool
    quality: float
    fallback_reason: str
    detected_points: int
    tracked_points: int
    inliers: int
    inlier_ratio: float
    reprojection_error: float
    coverage: float
    elapsed_ms: float

    def to_dict(self):
        payload = asdict(self)
        payload["homography"] = self.homography.reshape(-1).tolist()
        return payload


class CameraMotionEstimator:
    def __init__(self, config=None):
        self.config = config or CameraMotionConfig()

    @staticmethod
    def _fallback(reason, start, detected=0, tracked=0, inliers=0,
                  inlier_ratio=0.0, reprojection_error=float("inf"), coverage=0.0):
        return CameraMotionResult(
            homography=np.eye(3, dtype=np.float64), valid=False, quality=0.0,
            fallback_reason=reason, detected_points=int(detected),
            tracked_points=int(tracked), inliers=int(inliers),
            inlier_ratio=float(inlier_ratio),
            reprojection_error=float(reprojection_error), coverage=float(coverage),
            elapsed_ms=(perf_counter() - start) * 1000.0)

    @staticmethod
    def _coverage(points, width, height):
        if len(points) < 3 or width <= 0 or height <= 0:
            return 0.0
        hull = cv2.convexHull(np.asarray(points, dtype=np.float32).reshape(-1, 1, 2))
        return float(cv2.contourArea(hull) / float(width * height))

    def estimate(self, previous_rgb, current_rgb, target_box=None):
        start = perf_counter()
        if previous_rgb is None or current_rgb is None:
            return self._fallback("missing_frame", start)
        if previous_rgb.shape[:2] != current_rgb.shape[:2]:
            return self._fallback("frame_shape_changed", start)

        previous_gray = cv2.cvtColor(previous_rgb, cv2.COLOR_RGB2GRAY)
        current_gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY)
        height, width = previous_gray.shape
        mask = np.full_like(previous_gray, 255, dtype=np.uint8)
        if target_box is not None:
            x, y, w, h = [float(value) for value in target_box]
            margin_x = self.config.exclusion_margin * w
            margin_y = self.config.exclusion_margin * h
            x1 = int(np.clip(np.floor(x - margin_x), 0, width))
            y1 = int(np.clip(np.floor(y - margin_y), 0, height))
            x2 = int(np.clip(np.ceil(x + w + margin_x), 0, width))
            y2 = int(np.clip(np.ceil(y + h + margin_y), 0, height))
            mask[y1:y2, x1:x2] = 0

        points = cv2.goodFeaturesToTrack(
            previous_gray, maxCorners=self.config.max_corners,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.min_distance, mask=mask,
            blockSize=self.config.block_size)
        detected = 0 if points is None else len(points)
        if points is None or detected < self.config.min_correspondences:
            return self._fallback("insufficient_features", start, detected=detected)

        tracked_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if tracked_points is None or status is None:
            return self._fallback("optical_flow_failed", start, detected=detected)
        backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray, previous_gray, tracked_points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if backward_points is None or backward_status is None:
            return self._fallback("backward_flow_failed", start, detected=detected)
        forward_backward_error = np.linalg.norm(
            backward_points.reshape(-1, 2) - points.reshape(-1, 2), axis=1)
        status = (
            status.reshape(-1).astype(bool) &
            backward_status.reshape(-1).astype(bool) &
            (forward_backward_error <= self.config.max_forward_backward_error))
        source = points.reshape(-1, 2)[status]
        target = tracked_points.reshape(-1, 2)[status]
        finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
        source, target = source[finite], target[finite]
        tracked = len(source)
        if tracked < self.config.min_correspondences:
            return self._fallback(
                "insufficient_tracks", start, detected=detected, tracked=tracked)

        homography, inlier_mask = cv2.findHomography(
            source, target, method=cv2.RANSAC,
            ransacReprojThreshold=self.config.ransac_threshold,
            maxIters=2000, confidence=0.995)
        if homography is None or inlier_mask is None:
            return self._fallback(
                "homography_failed", start, detected=detected, tracked=tracked)
        homography = np.asarray(homography, dtype=np.float64)
        if not np.all(np.isfinite(homography)) or abs(homography[2, 2]) < 1e-8:
            return self._fallback(
                "invalid_homography", start, detected=detected, tracked=tracked)
        homography /= homography[2, 2]

        frame_corners = np.array(
            [[[0.0, 0.0]], [[float(width), 0.0]],
             [[float(width), float(height)]], [[0.0, float(height)]]],
            dtype=np.float32)
        warped_corners = cv2.perspectiveTransform(frame_corners, homography).reshape(-1, 2)
        warped_area = abs(float(cv2.contourArea(warped_corners.astype(np.float32))))
        area_ratio = warped_area / float(width * height)
        coordinate_limit = 3.0 * max(width, height)
        if (not np.isfinite(warped_corners).all() or
                np.max(np.abs(warped_corners)) > coordinate_limit or
                not self.config.min_frame_area_ratio <= area_ratio <= self.config.max_frame_area_ratio):
            return self._fallback(
                "implausible_frame_warp", start, detected=detected, tracked=tracked)

        inlier_mask = inlier_mask.reshape(-1).astype(bool)
        inliers = int(inlier_mask.sum())
        ratio = float(inliers / max(tracked, 1))
        projected = cv2.perspectiveTransform(
            source.reshape(-1, 1, 2), homography).reshape(-1, 2)
        errors = np.linalg.norm(projected - target, axis=1)
        reprojection = float(np.median(errors[inlier_mask])) if inliers else float("inf")
        coverage = min(
            self._coverage(source[inlier_mask], width, height),
            self._coverage(target[inlier_mask], width, height)) if inliers else 0.0

        failure = None
        if inliers < self.config.min_correspondences:
            failure = "insufficient_inliers"
        elif ratio < self.config.min_inlier_ratio:
            failure = "low_inlier_ratio"
        elif reprojection > self.config.max_reprojection_error:
            failure = "high_reprojection_error"
        elif coverage < self.config.min_coverage:
            failure = "low_spatial_coverage"
        if failure:
            return self._fallback(
                failure, start, detected, tracked, inliers, ratio,
                reprojection, coverage)

        ratio_quality = np.clip(
            (ratio - self.config.min_inlier_ratio) /
            max(1.0 - self.config.min_inlier_ratio, 1e-6), 0.0, 1.0)
        reprojection_quality = np.clip(
            1.0 - reprojection / self.config.max_reprojection_error, 0.0, 1.0)
        coverage_quality = np.clip(
            coverage / max(4.0 * self.config.min_coverage, 1e-6), 0.0, 1.0)
        quality = float(
            (max(ratio_quality, 1e-6) * max(reprojection_quality, 1e-6) *
             max(coverage_quality, 1e-6)) ** (1.0 / 3.0))
        return CameraMotionResult(
            homography=homography, valid=True, quality=quality,
            fallback_reason="", detected_points=detected, tracked_points=tracked,
            inliers=inliers, inlier_ratio=ratio,
            reprojection_error=reprojection, coverage=coverage,
            elapsed_ms=(perf_counter() - start) * 1000.0)
