from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

CONFIGS = Path(__file__).parents[1] / "configs"


def load(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


def test_cnn_initialization_control_changes_only_pretraining() -> None:
    pretrained = load("deeplabv3_resnet50_binary_pretrained.yaml")
    scratch = load("deeplabv3_resnet50_binary_scratch.yaml")
    assert pretrained["data"] == scratch["data"]
    assert pretrained["training"] == scratch["training"]
    pretrained_model = deepcopy(pretrained["model"])
    scratch_model = deepcopy(scratch["model"])
    assert pretrained_model.pop("pretrained") is True
    assert scratch_model.pop("pretrained") is False
    assert pretrained_model == scratch_model


def test_resolution_controls_keep_batch_and_optimization_fixed() -> None:
    names = {
        384: "dinov3_vitb16_binary_multilayer_conv_cosine.yaml",
        512: "dinov3_vitb16_binary_multilayer_conv_cosine_512.yaml",
        768: "dinov3_vitb16_binary_multilayer_conv_cosine_768.yaml",
    }
    configs = {size: load(name) for size, name in names.items()}
    reference = configs[384]
    for size, config in configs.items():
        assert config["data"]["image_size"] == size
        assert config["data"]["batch_size"] == 16
        assert config["training"] == reference["training"]
        assert config["model"] == reference["model"]
