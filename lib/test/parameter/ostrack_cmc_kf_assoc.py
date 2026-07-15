"""Parameters for the E1-E6 inference-only CMC-KF association chain."""

from pathlib import Path

from lib.config.ostrack.config import cfg, update_config_from_file
from lib.test.evaluation.environment import env_settings
from lib.test.parameter.ostrack_checkpoint_registry import (
    parse_parameter_name,
    resolve_paths,
    verify_checkpoint,
)
from lib.test.utils import TrackerParams


VARIANTS_V1 = {
    "e0": {
        "cmc_enabled": False, "kf_enabled": False,
        "association_backend": "appearance_top1",
        "cmc_quality_gate": False, "kf_quality_gate": False,
        "appearance_ambiguity_gate": False, "observation_rejection": False,
    },
    "e1": {
        "cmc_enabled": True, "kf_enabled": False,
        "association_backend": "appearance_top1",
        "cmc_quality_gate": False, "kf_quality_gate": False,
        "appearance_ambiguity_gate": False, "observation_rejection": False,
    },
    "e2": {
        "cmc_enabled": False, "kf_enabled": True,
        "association_backend": "mbpp_product",
        "cmc_quality_gate": False, "kf_quality_gate": False,
        "appearance_ambiguity_gate": False, "observation_rejection": False,
    },
    "e3": {
        "cmc_enabled": True, "kf_enabled": True,
        "association_backend": "mbpp_product",
        "cmc_quality_gate": False, "kf_quality_gate": False,
        "appearance_ambiguity_gate": False, "observation_rejection": False,
    },
    "e4": {
        "cmc_enabled": True, "kf_enabled": True,
        "association_backend": "reliability_iou",
        "cmc_quality_gate": True, "kf_quality_gate": False,
        "appearance_ambiguity_gate": False, "observation_rejection": False,
    },
    "e5": {
        "cmc_enabled": True, "kf_enabled": True,
        "association_backend": "reliability_iou",
        "cmc_quality_gate": True, "kf_quality_gate": True,
        "appearance_ambiguity_gate": False, "observation_rejection": False,
    },
    "e6": {
        "cmc_enabled": True, "kf_enabled": True,
        "association_backend": "reliability_iou",
        "cmc_quality_gate": True, "kf_quality_gate": True,
        "appearance_ambiguity_gate": True, "observation_rejection": True,
    },
}


VARIANTS_V2 = {
    key: dict(value) for key, value in VARIANTS_V1.items()
}
for _variant in ("e2", "e3"):
    VARIANTS_V2[_variant]["association_backend"] = "fixed_iou"

VARIANTS_V2.update({
    "m1": {
        "cmc_enabled": True, "kf_enabled": True,
        "association_backend": "appearance_top1",
        "cmc_quality_gate": False, "kf_quality_gate": False,
        "appearance_ambiguity_gate": False, "observation_rejection": False,
    },
    "m3": {
        "cmc_enabled": True, "kf_enabled": True,
        "association_backend": "fixed_iou",
        "cmc_quality_gate": False, "kf_quality_gate": False,
        "appearance_ambiguity_gate": False, "observation_rejection": True,
    },
})

VARIANT_CONFIGS = {"v1": VARIANTS_V1, "v2": VARIANTS_V2}
VARIANT_CONFIGS["v3"] = {
    key: dict(value) for key, value in VARIANTS_V2.items()
}
VARIANT_CONFIGS["v4"] = {
    key: dict(value) for key, value in VARIANT_CONFIGS["v3"].items()
}
VARIANT_CONFIGS["v5"] = {
    key: dict(value) for key, value in VARIANT_CONFIGS["v4"].items()
}
VARIANT_CONFIGS["v6"] = {
    key: dict(value) for key, value in VARIANT_CONFIGS["v5"].items()
}
VARIANT_CONFIGS["v6"]["n2"] = {
    **VARIANT_CONFIGS["v5"]["e6"],
    "switch_control_enabled": True,
}


def parameters(parameter_name: str):
    params = TrackerParams()
    environment = env_settings()
    spec = parse_parameter_name(
        parameter_name, expected_variant=None, expected_version=None,
        allowed_variants=set().union(
            *(variant_config.keys() for variant_config in VARIANT_CONFIGS.values())),
        allowed_versions=VARIANT_CONFIGS)
    if spec.variant not in VARIANT_CONFIGS[spec.version]:
        raise ValueError(
            f"variant {spec.variant!r} is not defined for config version {spec.version!r}")
    yaml_file, checkpoint_file = resolve_paths(
        spec, environment.prj_dir, environment.save_dir)
    verify_checkpoint(spec, checkpoint_file)
    update_config_from_file(str(yaml_file))

    params.cfg = cfg
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.checkpoint = str(checkpoint_file)
    params.checkpoint_id = spec.checkpoint_id
    params.checkpoint_sha256 = spec.sha256
    params.config_name = spec.config_name
    params.parameter_name = parameter_name
    params.variant = spec.variant
    params.config_version = spec.version
    params.variant_config = dict(VARIANT_CONFIGS[spec.version][spec.variant])
    params.appearance_score_source = (
        "raw" if params.variant_config["association_backend"] == "mbpp_product"
        or spec.version in {"v1", "v2"} else "hann")
    params.observation_policy = (
        "clear_top1_or_consistent"
        if spec.version in {"v4", "v5", "v6"} else "legacy_absolute")
    params.candidate_top_k = 5
    params.mbpp_proposal_count = 30
    params.mbpp_nms_iou = 0.8
    params.fixed_motion_weight = 0.25
    params.maximum_motion_weight = 0.25
    params.mbpp_keep_iou = 0.60
    params.ambiguity_margin = 0.15
    params.observation_min_score = 0.15
    params.observation_strong_score = 0.25
    params.innovation_chi2_threshold = 13.2767
    params.switch_minimum_final_margin = 0.02
    params.switch_confirmation_frames = 2
    params.switch_minimum_consistency_iou = 0.30
    params.diagnostic_root = str(
        Path(environment.save_dir) / "cmc_kf_candidate" / "runtime")
    params.save_all_boxes = False
    params.debug = 0
    return params
