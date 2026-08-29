from __future__ import annotations

from types import SimpleNamespace

from PIL import Image
from torchvision.transforms import functional as TF
from transformers import AutoImageProcessor

from .config import ModelConfig


class ImageNetProcessor:
    """Minimal HF-compatible processor for torchvision ImageNet encoders."""

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_mean = mean
    image_std = std

    def __call__(
        self, images: Image.Image, return_tensors: str = "pt", do_resize: bool = False
    ) -> SimpleNamespace:
        if return_tensors != "pt":
            raise ValueError("ImageNetProcessor supports only return_tensors='pt'")
        if do_resize:
            raise ValueError("Images must be resized together with masks by PlantSegDataset")
        pixels = TF.normalize(TF.to_tensor(images), self.mean, self.std)
        return SimpleNamespace(pixel_values=pixels.unsqueeze(0))


def build_processor(config: ModelConfig):
    if config.architecture == "dinov3":
        return AutoImageProcessor.from_pretrained(config.backbone)
    if config.architecture == "deeplabv3_resnet50":
        return ImageNetProcessor()
    raise ValueError(f"Unknown model architecture: {config.architecture}")
