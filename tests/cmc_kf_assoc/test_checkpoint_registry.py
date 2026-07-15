import pytest

from lib.test.parameter.ostrack_checkpoint_registry import parse_parameter_name
from tools.cmc_kf_assoc.run_experiment import validate_declared_identity


def test_parameter_identity_is_strict_and_explicit():
    spec = parse_parameter_name("e3__official_vitb256_ce_ep300__v1", expected_variant=None,
                                allowed_variants={"e1", "e2", "e3"})
    assert spec.variant == "e3"
    assert spec.checkpoint_id == "official_vitb256_ce_ep300"


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        parse_parameter_name("e9__official_vitb256_ce_ep300__v1", expected_variant=None,
                             allowed_variants={"e1", "e2", "e3"})


def test_version_set_is_explicit():
    spec = parse_parameter_name(
        "e3__official_vitb256_ce_ep300__v2", expected_variant=None,
        expected_version=None, allowed_variants={"e3"}, allowed_versions={"v1", "v2"})
    assert spec.version == "v2"


def test_v2_ablation_keeps_e2_e3_formula_fixed():
    from lib.test.parameter.ostrack_cmc_kf_assoc import VARIANT_CONFIGS
    assert VARIANT_CONFIGS["v2"]["e2"]["association_backend"] == "fixed_iou"
    assert VARIANT_CONFIGS["v2"]["e3"]["association_backend"] == "fixed_iou"
    assert VARIANT_CONFIGS["v2"]["e4"]["association_backend"] == "reliability_iou"
    assert VARIANT_CONFIGS["v2"]["m1"]["association_backend"] == "appearance_top1"
    assert VARIANT_CONFIGS["v2"]["m3"]["observation_rejection"] is True
    assert VARIANT_CONFIGS["v3"]["e3"]["association_backend"] == "fixed_iou"
    assert VARIANT_CONFIGS["v4"]["e6"]["observation_rejection"] is True
    assert VARIANT_CONFIGS["v5"]["e6"]["observation_rejection"] is True
    assert VARIANT_CONFIGS["v6"]["n2"]["switch_control_enabled"] is True


def test_experiment_identity_must_match_registry():
    validate_declared_identity(
        "E6", "e6__official_vitb256_ce_ep300__v5",
        "official_vitb256_ce_ep300",
        "8e01d6251569ac84dbb53fdd062cd938551be2d889582f789aa3070196f42f41")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_declared_identity(
            "E6", "e6__official_vitb256_ce_ep300__v5",
            "official_vitb256_ce_ep300", "0" * 64)


def test_experiment_variant_must_match_parameter():
    with pytest.raises(ValueError, match="variant mismatch"):
        validate_declared_identity(
            "E3", "e6__official_vitb256_ce_ep300__v5",
            "official_vitb256_ce_ep300",
            "8e01d6251569ac84dbb53fdd062cd938551be2d889582f789aa3070196f42f41")
