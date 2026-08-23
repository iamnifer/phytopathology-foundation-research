from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def resize_pair(
    image: Image.Image, mask: Image.Image, size: int, mode: str
) -> tuple[Image.Image, Image.Image]:
    target_size = (size, size)
    if mode == "stretch":
        return (
            image.resize(target_size, Image.Resampling.BILINEAR),
            mask.resize(target_size, Image.Resampling.NEAREST),
        )
    if mode == "pad":
        return (
            ImageOps.pad(image, target_size, Image.Resampling.BILINEAR, color=0),
            ImageOps.pad(mask, target_size, Image.Resampling.NEAREST, color=0),
        )
    raise ValueError("resize_mode must be 'stretch' or 'pad'")


def augment_pair(
    image: Image.Image, mask: Image.Image, preset: str
) -> tuple[Image.Image, Image.Image]:
    if preset not in {"none", "horizontal_flip", "spatial_color_light"}:
        raise ValueError(
            "augmentation must be 'none', 'horizontal_flip', or 'spatial_color_light'"
        )
    if preset == "none":
        return image, mask
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if preset == "horizontal_flip":
        return image, mask

    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if random.random() < 0.5:
        turns = random.randint(1, 3)
        image = image.rotate(90 * turns)
        mask = mask.rotate(90 * turns)
    if random.random() < 0.4:
        angle = random.uniform(-30, 30)
        translate = [
            int(random.uniform(-0.1, 0.1) * image.width),
            int(random.uniform(-0.1, 0.1) * image.height),
        ]
        scale = random.uniform(0.9, 1.1)
        image = TF.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        mask = TF.affine(
            mask,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
    if random.random() < 0.5:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.4:
        image = ImageEnhance.Color(image).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.2:
        array = np.asarray(image, dtype=np.float32)
        noise = np.random.normal(0.0, random.uniform(5.0, 15.0), array.shape)
        image = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))
    if random.random() < 0.15:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    return image, mask


class PlantSegDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lazy PlantSeg loader; unlike the legacy notebook, it does not cache images in RAM."""

    SPLIT_NAMES = {"train": "Training", "val": "Validation", "test": "Test"}
    DIRECTORY_NAMES = {"train": "train", "val": "val", "test": "test"}

    def __init__(
        self,
        root: str | Path,
        split: str,
        processor,
        image_size: int,
        num_classes: int = 116,
        ignore_index: int = 255,
        metadata_file: str = "Metadatav2.csv",
        augment: bool = False,
        binary_masks: bool = False,
        resize_mode: str = "stretch",
        augmentation: str = "horizontal_flip",
        limit: int | None = None,
    ) -> None:
        split = split.lower()
        if split not in self.SPLIT_NAMES:
            raise ValueError(f"Unknown split {split!r}; expected one of {tuple(self.SPLIT_NAMES)}")
        if image_size % 16:
            raise ValueError("image_size must be divisible by the DINOv3 patch size (16)")

        self.root = Path(root)
        self.split = split
        self.processor = processor
        self.image_size = image_size
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.augment = augment
        self.binary_masks = binary_masks
        self.resize_mode = resize_mode
        self.augmentation = augmentation
        self._foreground_fractions: np.ndarray | None = None

        metadata_path = self.root / metadata_file
        if not metadata_path.is_file():
            raise FileNotFoundError(f"PlantSeg metadata not found: {metadata_path}")
        frame = pd.read_csv(metadata_path)
        required = {"Name", "Label file", "Split"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing metadata columns: {sorted(missing)}")
        frame = frame[frame["Split"] == self.SPLIT_NAMES[split]].reset_index(drop=True)
        if limit is not None:
            frame = frame.iloc[:limit].copy()
        if frame.empty:
            raise ValueError(f"No samples found for split={split!r} in {metadata_path}")
        self.samples = list(zip(frame["Name"], frame["Label file"], strict=True))

        directory = self.DIRECTORY_NAMES[split]
        if split == "val" and not (self.root / "images" / directory).is_dir():
            directory = "validation"
        self.image_dir = self.root / "images" / directory
        self.mask_dir = self.root / "annotations" / directory

    def __len__(self) -> int:
        return len(self.samples)

    def foreground_fractions(self) -> np.ndarray:
        """Return foreground fractions in original masks, excluding ignored pixels."""
        if not self.binary_masks:
            raise ValueError("foreground fractions are defined only for binary masks")
        if self._foreground_fractions is None:
            fractions = []
            for _, mask_name in self.samples:
                mask_path = self.mask_dir / str(mask_name)
                with Image.open(mask_path) as source:
                    target = np.asarray(source.convert("L"))
                valid = target != self.ignore_index
                valid_count = int(valid.sum())
                fraction = float(((target > 0) & valid).sum() / valid_count) if valid_count else 0.0
                fractions.append(fraction)
            self._foreground_fractions = np.asarray(fractions, dtype=np.float64)
        return self._foreground_fractions.copy()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_name, mask_name = self.samples[index]
        image_path = self.image_dir / str(image_name)
        mask_path = self.mask_dir / str(mask_name)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(mask_path) as source:
            mask = source.convert("L")

        image, mask = resize_pair(image, mask, self.image_size, self.resize_mode)
        if self.augment:
            image, mask = augment_pair(image, mask, self.augmentation)

        pixels = self.processor(
            images=image,
            return_tensors="pt",
            do_resize=False,
        ).pixel_values.squeeze(0)
        target = torch.from_numpy(np.array(mask, dtype=np.int64, copy=True))
        if self.binary_masks:
            ignored = target == self.ignore_index
            target = (target > 0).to(torch.int64)
            target[ignored] = self.ignore_index
        valid = target != self.ignore_index
        if valid.any():
            minimum, maximum = int(target[valid].min()), int(target[valid].max())
            if minimum < 0 or maximum >= self.num_classes:
                raise ValueError(
                    f"Mask {mask_path} contains labels [{minimum}, {maximum}], "
                    f"but num_classes={self.num_classes}"
                )
        return pixels, target
