from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_name, mask_name = self.samples[index]
        image_path = self.image_dir / str(image_name)
        mask_path = self.mask_dir / str(mask_name)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(mask_path) as source:
            mask = source.convert("L")

        size = (self.image_size, self.image_size)
        image = image.resize(size, Image.Resampling.BILINEAR)
        mask = mask.resize(size, Image.Resampling.NEAREST)
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        pixels = self.processor(
            images=image,
            return_tensors="pt",
            do_resize=False,
        ).pixel_values.squeeze(0)
        target = torch.from_numpy(np.array(mask, dtype=np.int64, copy=True))
        valid = target != self.ignore_index
        if valid.any():
            minimum, maximum = int(target[valid].min()), int(target[valid].max())
            if minimum < 0 or maximum >= self.num_classes:
                raise ValueError(
                    f"Mask {mask_path} contains labels [{minimum}, {maximum}], "
                    f"but num_classes={self.num_classes}"
                )
        return pixels, target
