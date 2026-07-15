from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_splits_have_expected_counts_and_no_overlap():
    development = yaml.safe_load(
        (ROOT / "experiments/cmc_kf_candidate/splits/development_v1.yaml").read_text(encoding="utf-8"))
    holdout = yaml.safe_load(
        (ROOT / "experiments/cmc_kf_candidate/splits/holdout_v1.yaml").read_text(encoding="utf-8"))
    assert development["total_sequences"] == 20
    assert holdout["total_sequences"] == 20
    for dataset in ("dtb70", "uavdt", "visdrone", "uav123"):
        dev_names = development["datasets"][dataset]
        holdout_names = holdout["datasets"][dataset]
        assert len(dev_names) == len(set(dev_names)) == 5
        assert len(holdout_names) == len(set(holdout_names)) == 5
        assert set(dev_names).isdisjoint(holdout_names)
    assert len({name for names in development["datasets"].values() for name in names}) == 20
    assert len({name for names in holdout["datasets"].values() for name in names}) == 20


def test_n2_development_split_is_frozen_and_disjoint_from_v1_and_holdout():
    split_root = ROOT / "experiments/cmc_kf_candidate/splits"
    development_v1 = yaml.safe_load(
        (split_root / "development_v1.yaml").read_text(encoding="utf-8"))
    development_v2 = yaml.safe_load(
        (split_root / "development_v2.yaml").read_text(encoding="utf-8"))
    holdout = yaml.safe_load(
        (split_root / "holdout_v1.yaml").read_text(encoding="utf-8"))
    assert development_v2["result_blind"] is True
    assert development_v2["derived_before_n2_metrics"] is True
    assert development_v2["total_sequences"] == 20
    assert development_v2["total_frames"] == 14941
    for dataset in ("dtb70", "uavdt", "visdrone", "uav123"):
        names = development_v2["datasets"][dataset]
        assert len(names) == len(set(names)) == 5
        assert set(names).isdisjoint(development_v1["datasets"][dataset])
        assert set(names).isdisjoint(holdout["datasets"][dataset])
