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
