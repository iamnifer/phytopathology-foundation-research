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


def test_imagenet_processor_shape() -> None:
    from PIL import Image

    output = ImageNetProcessor()(Image.new("RGB", (32, 48)), return_tensors="pt")
    assert output.pixel_values.shape == (1, 3, 48, 32)
