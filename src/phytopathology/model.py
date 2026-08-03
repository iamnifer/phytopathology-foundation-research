from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


class ConvDecoder(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )


class DINOv3Segmenter(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        decoder: str = "linear",
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.freeze_backbone = freeze_backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(not freeze_backbone)

        hidden_size = self.backbone.config.hidden_size
        if decoder == "linear":
            self.decoder = nn.Conv2d(hidden_size, num_classes, kernel_size=1)
        elif decoder == "conv":
            self.decoder = ConvDecoder(hidden_size, num_classes)
        else:
            raise ValueError("decoder must be 'linear' or 'conv'")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        height, width = pixel_values.shape[-2:]
        patch_size = self.backbone.config.patch_size
        if isinstance(patch_size, (tuple, list)):
            patch_height, patch_width = patch_size
        else:
            patch_height = patch_width = patch_size
        if height % patch_height or width % patch_width:
            raise ValueError("Input dimensions must be divisible by the backbone patch size")

        grad_enabled = torch.is_grad_enabled() and not self.freeze_backbone
        with torch.set_grad_enabled(grad_enabled):
            tokens = self.backbone(pixel_values=pixel_values).last_hidden_state
        num_register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 0))
        patches = tokens[:, 1 + num_register_tokens :, :]
        grid = (height // patch_height, width // patch_width)
        expected_patches = grid[0] * grid[1]
        if patches.shape[1] != expected_patches:
            raise RuntimeError(
                f"Expected {expected_patches} patch tokens for grid {grid}, got {patches.shape[1]}"
            )
        features = patches.transpose(1, 2).unflatten(2, grid)
        logits = self.decoder(features)
        return F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
