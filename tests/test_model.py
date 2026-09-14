from __future__ import annotations

import torch

from phytopathology.config import ModelConfig
from phytopathology.model import build_model, trainable_state_dict
from phytopathology.processing import ImageNetProcessor


def test_deeplab_factory_without_download() -> None:
    config = ModelConfig(architecture="deeplabv3_resnet50", pretrained=False)
    model = build_model(config, num_classes=2).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 64, 64))
    assert output.shape == (1, 2, 64, 64)
    assert trainable_state_dict(model).keys() == model.state_dict().keys()


def test_frozen_resnet_linear_probe_without_download() -> None:
    config = ModelConfig(
        architecture="resnet50_linear_probe",
        pretrained=False,
        feature_grid_size=4,
    )
    model = build_model(config, num_classes=2).train()
    assert not model.backbone.training
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    assert trainable_parameters == 4098
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 64, 64))
    assert output.shape == (1, 2, 64, 64)
    assert set(trainable_state_dict(model)) == {"decoder.weight", "decoder.bias"}


def test_imagenet_processor_shape() -> None:
    from PIL import Image

    output = ImageNetProcessor()(Image.new("RGB", (32, 48)), return_tensors="pt")
    assert output.pixel_values.shape == (1, 3, 48, 32)
