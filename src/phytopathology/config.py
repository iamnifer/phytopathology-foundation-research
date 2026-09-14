from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "dinov3_baseline"
    output_dir: str = "runs"
    seed: int = 42


@dataclass(frozen=True)
class DataConfig:
    root: str = "data/plantsegv2"
    metadata_file: str = "Metadatav2.csv"
    image_size: int = 384
    num_classes: int = 116
    ignore_index: int = 255
    batch_size: int = 16
    num_workers: int = 8
    binary_masks: bool = False
    resize_mode: str = "stretch"
    augmentation: str = "horizontal_flip"
    sampling: str = "shuffle"
    small_lesion_quantile: float = 0.25
    small_lesion_factor: float = 2.0
    subset_file: str | None = None
    epoch_samples: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "dinov3"
    backbone: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    decoder: str = "linear"
    freeze_backbone: bool = True
    feature_layers: str = "last"
    pretrained: bool = True
    feature_grid_size: int = 24


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    amp: bool = True
    exclude_background_from_miou: bool = False
    selection_metric: str = "miou"
    scheduler: str = "constant"
    minimum_learning_rate: float = 1e-5


@dataclass(frozen=True)
class Config:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> Config:
    with Path(path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    return Config(
        experiment=ExperimentConfig(**raw.get("experiment", {})),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
    )
