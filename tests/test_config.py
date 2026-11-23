from pathlib import Path

import pytest
from omegaconf.errors import ConfigKeyError, ValidationError

from ptbxl_forecast.config import config_hash, load_config

BASE = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


def test_base_config_and_overrides():
    cfg = load_config([BASE], ["seed=123", "data.horizon=100"])
    assert cfg.seed == 123 and cfg.data.horizon == 100 and cfg.data.history == 700


def test_adopted_pipeline_comes_from_config():
    from ptbxl_forecast.preprocessing import Pipeline

    pipe = Pipeline.from_config(load_config([BASE]))
    assert (pipe.filter_name, pipe.in_channels, pipe.out_leads) == ("decompose_0.67", 24, 12)


@pytest.mark.parametrize(
    "overrides, error",
    [
        (["data.horizn=100"], ConfigKeyError),  # typo
        (["seed=abc"], ValidationError),
        (["data.history=7.5"], ValidationError),
        (["data.val_folds=[8]"], ValueError),  # overlaps training folds
        (["data.test_folds=[11]"], ValueError),
        (["data.history=800"], ValueError),  # 800 + 290 > 1000 - 10
        (["data.trim_end=0", "data.horizon=301"], ValueError),
        (["runtime.device=gpu"], ValueError),
        (["preprocessing.filter=bandpass"], ValueError),
    ],
)
def test_invalid_configs_are_rejected(overrides, error):
    with pytest.raises(error):
        load_config([BASE], overrides)


def test_unknown_key_in_file_is_rejected(tmp_path):
    extra = tmp_path / "typo.yaml"
    extra.write_text("data:\n  horizn: 100\n")
    with pytest.raises(ConfigKeyError):
        load_config([BASE, extra])


def test_config_hash_tracks_science_only(tmp_path):
    reordered = tmp_path / "reordered.yaml"
    reordered.write_text("data:\n  horizon: 290\n  history: 700\nseed: 42\n")
    base = config_hash(load_config([BASE]))
    assert config_hash(load_config([BASE, reordered])) == base
    machine_specific = ["paths.data=/elsewhere", "run_name=x", "tracking.mlflow=true"]
    assert config_hash(load_config([BASE], machine_specific)) == base
    assert config_hash(load_config([BASE], ["seed=43"])) != base
    assert config_hash(load_config([BASE], ["data.allow_test=true"])) != base
