"""Single-forward OSTrack with inference-only CMC, KF, and Top-K association."""

from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from lib.test.tracker.cmc_kf_assoc.association import (
    AssociationConfig,
    ambiguity_quality,
    associate_candidates,
)
from lib.test.tracker.cmc_kf_assoc.camera_motion import CameraMotionEstimator
from lib.test.tracker.cmc_kf_assoc.diagnostics import JsonlDiagnostics
from lib.test.tracker.cmc_kf_assoc.kalman_box import KalmanBoxFilter
from lib.test.tracker.cmc_kf_assoc.observation import decide_observation
from lib.test.tracker.cmc_kf_assoc.response_candidates import extract_mbpp_candidates
from lib.test.tracker.cmc_kf_assoc.state_propagation import project_box
from lib.test.tracker.cmc_kf_assoc.switch_control import (
    CameraConsistentSwitchController,
    SwitchControlConfig,
)
from lib.test.tracker.ostrack import OSTrack as BaselineOSTrack
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box


def _safe_component(value):
    return "".join(character if character.isalnum() or character in "-_." else "_"
                   for character in str(value))


class OSTrackCMCKFAssoc(BaselineOSTrack):
    def __init__(self, params, dataset_name):
        super().__init__(params, dataset_name)
        self.variant_config = dict(params.variant_config)
        self.camera_motion = CameraMotionEstimator()
        self.kalman = None
        self.previous_frame = None
        self.previous_output_box = None
        self.association_config = AssociationConfig(
            backend=self.variant_config["association_backend"],
            fixed_motion_weight=params.fixed_motion_weight,
            maximum_motion_weight=params.maximum_motion_weight,
            mbpp_keep_iou=params.mbpp_keep_iou,
            ambiguity_margin=params.ambiguity_margin)
        self.switch_controller = None
        if self.variant_config.get("switch_control_enabled", False):
            self.switch_controller = CameraConsistentSwitchController(
                SwitchControlConfig(
                    minimum_final_margin=params.switch_minimum_final_margin,
                    confirmation_frames=params.switch_confirmation_frames,
                    minimum_consistency_iou=params.switch_minimum_consistency_iou,
                    require_valid_cmc=True))

        diagnostic_path = (
            Path(params.diagnostic_root) /
            _safe_component(params.parameter_name) /
            _safe_component(dataset_name) /
            f"{_safe_component(getattr(params, 'sequence_name', 'unknown'))}.jsonl")
        self.diagnostics = JsonlDiagnostics(diagnostic_path)
        self.diagnostic_path = str(diagnostic_path)

    def initialize(self, image, info: dict):
        output = super().initialize(image, info)
        self.previous_frame = image.copy()
        self.previous_output_box = [float(value) for value in info["init_bbox"]]
        if self.variant_config["kf_enabled"]:
            self.kalman = KalmanBoxFilter(self.previous_output_box)
        self.diagnostics.write({
            "frame_id": 0,
            "event": "initialize",
            "variant": self.params.variant,
            "checkpoint_id": self.params.checkpoint_id,
            "target_bbox": self.previous_output_box,
        })
        return output

    @staticmethod
    def _map_candidate_to_image(candidate, resize_factor, reference_box, search_size):
        cx_ref = reference_box[0] + 0.5 * reference_box[2]
        cy_ref = reference_box[1] + 0.5 * reference_box[3]
        cx, cy, width, height = candidate.normalized_box
        cx *= search_size / resize_factor
        cy *= search_size / resize_factor
        width *= search_size / resize_factor
        height *= search_size / resize_factor
        half_side = 0.5 * search_size / resize_factor
        cx += cx_ref - half_side
        cy += cy_ref - half_side
        return [cx - 0.5 * width, cy - 0.5 * height, width, height]

    @staticmethod
    def _response_apce(response):
        values = response.detach().float()
        minimum = values.min()
        peak = values.max()
        denominator = torch.mean((values - minimum) ** 2)
        return float(((peak - minimum) ** 2 / (denominator + 1e-12)).item())

    def track(self, image, info: dict = None):
        frame_start = perf_counter()
        image_height, image_width, _ = image.shape
        self.frame_id += 1
        prior_output_box = list(self.previous_output_box)

        cmc_start = perf_counter()
        if self.variant_config["cmc_enabled"]:
            cmc = self.camera_motion.estimate(
                self.previous_frame, image, self.previous_output_box)
        else:
            cmc = None
        cmc_elapsed_ms = (perf_counter() - cmc_start) * 1000.0

        search_reference = list(self.previous_output_box)
        predicted_box = list(self.previous_output_box)
        propagation_jacobian = None
        if self.kalman is not None:
            if cmc is not None:
                try:
                    propagation_jacobian = self.kalman.propagate_camera(
                        cmc.homography, quality=cmc.quality if cmc.valid else 0.0)
                except (ValueError, np.linalg.LinAlgError) as error:
                    cmc.valid = False
                    cmc.quality = 0.0
                    cmc.fallback_reason = f"state_propagation_failed:{type(error).__name__}"
            predicted_box = self.kalman.predict()
            search_reference = list(predicted_box)
        elif cmc is not None and cmc.valid:
            try:
                search_reference = project_box(
                    self.previous_output_box, cmc.homography).tolist()
                predicted_box = list(search_reference)
            except ValueError as error:
                cmc.valid = False
                cmc.quality = 0.0
                cmc.fallback_reason = f"box_propagation_failed:{type(error).__name__}"
                search_reference = list(self.previous_output_box)
                predicted_box = list(search_reference)
        search_reference = clip_box(
            search_reference, image_height, image_width, margin=10)

        crop_start = perf_counter()
        search_patch, resize_factor, search_mask = sample_target(
            image, search_reference, self.params.search_factor,
            output_sz=self.params.search_size)
        search = self.preprocessor.process(search_patch, search_mask)
        crop_elapsed_ms = (perf_counter() - crop_start) * 1000.0

        network_start = perf_counter()
        with torch.no_grad():
            output = self.network.forward(
                template=self.z_dict1.tensors, search=search.tensors,
                ce_template_mask=self.box_mask_z)
        network_elapsed_ms = (perf_counter() - network_start) * 1000.0

        association_start = perf_counter()
        raw_response = output["score_map"]
        hann_response = self.output_window * raw_response
        candidates = extract_mbpp_candidates(
            hann_response, raw_response, output["size_map"], output["offset_map"],
            top_k=self.params.candidate_top_k,
            proposal_count=self.params.mbpp_proposal_count,
            nms_iou=self.params.mbpp_nms_iou)
        for candidate in candidates:
            candidate.score = (
                candidate.raw_score
                if self.params.appearance_score_source == "raw"
                else candidate.hann_score)
            mapped = self._map_candidate_to_image(
                candidate, resize_factor, search_reference, self.params.search_size)
            candidate.image_box = clip_box(
                mapped, image_height, image_width, margin=10)

        q_cmc = 1.0
        if self.variant_config["cmc_quality_gate"]:
            q_cmc = cmc.quality if cmc is not None and cmc.valid else 0.0
        q_kf = 1.0
        if self.variant_config["kf_quality_gate"] and self.kalman is not None:
            q_kf = self.kalman.quality
        # Observation-write reliability is independent of whether ambiguity
        # is enabled as an association-score gate.  Keeping one shared gated
        # value here made the M3 ablation reject clear Top-1 observations just
        # because its association gate was intentionally disabled.
        observation_q_ambiguity = ambiguity_quality(
            candidates, self.params.ambiguity_margin)
        association_q_ambiguity = (
            observation_q_ambiguity
            if self.variant_config["appearance_ambiguity_gate"] else 1.0)

        decision = associate_candidates(
            candidates, predicted_box, self.association_config,
            q_cmc=q_cmc, q_kf=q_kf, q_ambiguity=association_q_ambiguity)
        switch_decision = None
        if self.switch_controller is not None:
            switch_decision = self.switch_controller.decide(
                candidates, decision, cmc)
            decision.selected_index = switch_decision.selected_index
            decision.selected_rank = switch_decision.selected_rank
            decision.reason = f"n2_{switch_decision.reason}"
        selected = candidates[decision.selected_index]
        self.state = list(selected.image_box)

        innovation = None
        mahalanobis = None
        measurement_accepted = None
        rejection_reason = ""
        if self.kalman is not None:
            innovation, mahalanobis = self.kalman.innovation(self.state)
            measurement_accepted = True
            if self.variant_config["observation_rejection"]:
                observation = decide_observation(
                    selected.rank, selected.score, observation_q_ambiguity,
                    mahalanobis,
                    policy=self.params.observation_policy,
                    minimum_score=self.params.observation_min_score,
                    strong_score=self.params.observation_strong_score,
                    chi2_threshold=self.params.innovation_chi2_threshold)
                measurement_accepted = observation.accepted
                rejection_reason = "" if observation.accepted else observation.reason
            if measurement_accepted:
                self.kalman.update(self.state)
        association_elapsed_ms = (perf_counter() - association_start) * 1000.0

        self.previous_frame = image.copy()
        self.previous_output_box = list(self.state)
        response_flat = hann_response.detach().flatten()
        top_scores = torch.topk(response_flat, min(2, response_flat.numel())).values
        total_elapsed_ms = (perf_counter() - frame_start) * 1000.0

        self.diagnostics.write({
            "frame_id": self.frame_id,
            "variant": self.params.variant,
            "previous_output_box": prior_output_box,
            "search_reference_box": search_reference,
            "predicted_box": predicted_box,
            "cmc": None if cmc is None else cmc.to_dict(),
            "cmc_elapsed_ms": cmc_elapsed_ms,
            "propagation_jacobian": propagation_jacobian,
            "kf": None if self.kalman is None else {
                "mean": self.kalman.mean,
                "covariance_diagonal": np.diag(self.kalman.covariance),
                "position_std": self.kalman.position_std,
                "quality": self.kalman.quality,
                "age": self.kalman.age,
                "predict_only_count": self.kalman.predict_only_count,
                "innovation": innovation,
                "mahalanobis": mahalanobis,
            },
            "response": {
                "top1": float(top_scores[0].item()),
                "top2": float(top_scores[1].item()) if len(top_scores) > 1 else None,
                "top2_margin": float((top_scores[0] - top_scores[1]).item()) if len(top_scores) > 1 else None,
                "apce": self._response_apce(hann_response),
            },
            "candidates": [candidate.to_dict() for candidate in candidates],
            "association": decision.to_dict(),
            "switch_control": (
                None if switch_decision is None else switch_decision.to_dict()),
            "measurement_accepted": measurement_accepted,
            "rejection_reason": rejection_reason,
            "output_box": self.state,
            "timing_ms": {
                "crop": crop_elapsed_ms,
                "network": network_elapsed_ms,
                "association_and_update": association_elapsed_ms,
                "total": total_elapsed_ms,
            },
        })
        return {"target_bbox": self.state}


def get_tracker_class():
    return OSTrackCMCKFAssoc
