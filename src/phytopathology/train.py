from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoImageProcessor

from .config import Config, load_config
from .data import PlantSegDataset
from .metrics import BinaryAveragePrecision, SegmentationMetrics
from .model import DINOv3Segmenter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DINOv3 PlantSeg probe")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, help="Override data.root from YAML")
    parser.add_argument("--seed", type=int, help="Override experiment.seed from YAML")
    parser.add_argument("--smoke-test", action="store_true", help="Use 32 train and 16 val samples")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(config: Config) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(config.experiment.output_dir) / f"{timestamp}_{config.experiment.name}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def environment() -> dict[str, object]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    return {
        "git_commit": commit,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def small_lesion_sampling_weights(
    foreground_fractions: np.ndarray, quantile: float, factor: float
) -> tuple[torch.Tensor, float]:
    if not 0.0 < quantile < 1.0:
        raise ValueError("small_lesion_quantile must lie strictly between 0 and 1")
    if factor < 1.0:
        raise ValueError("small_lesion_factor must be at least 1")
    positive = foreground_fractions[foreground_fractions > 0]
    if positive.size == 0:
        raise ValueError("small-lesion sampling requires at least one foreground mask")
    threshold = float(np.quantile(positive, quantile))
    weights = np.ones(len(foreground_fractions), dtype=np.float64)
    small = (foreground_fractions > 0) & (foreground_fractions <= threshold)
    weights[small] = factor
    return torch.as_tensor(weights, dtype=torch.double), threshold


def make_loader(dataset, config: Config, shuffle: bool) -> DataLoader:
    sampler = None
    if shuffle and config.data.sampling == "small_lesion_oversample":
        weights, _ = small_lesion_sampling_weights(
            dataset.foreground_fractions(),
            config.data.small_lesion_quantile,
            config.data.small_lesion_factor,
        )
        generator = torch.Generator().manual_seed(config.experiment.seed)
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    elif shuffle and config.data.sampling != "shuffle":
        raise ValueError("data.sampling must be 'shuffle' or 'small_lesion_oversample'")
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.data.num_workers > 0,
    )


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def train_epoch(model, loader, optimizer, criterion, device, amp: bool) -> float:
    model.train()
    loss_sum = 0.0
    samples = 0
    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            logits = model(images)
            loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * images.shape[0]
        samples += images.shape[0]
    return loss_sum / samples


@torch.inference_mode()
def evaluate(model, loader, criterion, device, config: Config) -> dict[str, object]:
    model.eval()
    metrics = SegmentationMetrics(config.data.num_classes, config.data.ignore_index)
    average_precision = (
        BinaryAveragePrecision(ignore_index=config.data.ignore_index)
        if config.data.binary_masks
        else None
    )
    loss_sum = 0.0
    samples = 0
    for images, targets in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(device, config.training.amp):
            logits = model(images)
            loss = criterion(logits, targets)
        metrics.update(logits.argmax(dim=1), targets)
        if average_precision is not None:
            average_precision.update(logits.softmax(dim=1)[:, 1], targets)
        loss_sum += loss.item() * images.shape[0]
        samples += images.shape[0]
    result = metrics.compute(config.training.exclude_background_from_miou)
    if average_precision is not None:
        result["pixel_average_precision"] = average_precision.compute()
    result["loss"] = loss_sum / samples
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_root:
        object.__setattr__(config.data, "root", str(args.data_root))
    if args.seed is not None:
        object.__setattr__(config.experiment, "seed", args.seed)
    set_seed(config.experiment.seed)
    run_dir = make_run_dir(config)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config.as_dict(), sort_keys=False), encoding="utf-8"
    )
    (run_dir / "environment.json").write_text(
        json.dumps(environment(), indent=2), encoding="utf-8"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(config.model.backbone)
    limit_train, limit_val = (32, 16) if args.smoke_test else (None, None)
    train_set = PlantSegDataset(
        config.data.root,
        "train",
        processor,
        config.data.image_size,
        config.data.num_classes,
        config.data.ignore_index,
        config.data.metadata_file,
        augment=True,
        binary_masks=config.data.binary_masks,
        resize_mode=config.data.resize_mode,
        augmentation=config.data.augmentation,
        limit=limit_train,
    )
    val_set = PlantSegDataset(
        config.data.root,
        "val",
        processor,
        config.data.image_size,
        config.data.num_classes,
        config.data.ignore_index,
        config.data.metadata_file,
        binary_masks=config.data.binary_masks,
        resize_mode=config.data.resize_mode,
        limit=limit_val,
    )
    if config.data.sampling == "small_lesion_oversample":
        fractions = train_set.foreground_fractions()
        weights, threshold = small_lesion_sampling_weights(
            fractions,
            config.data.small_lesion_quantile,
            config.data.small_lesion_factor,
        )
        sampling_summary = {
            "method": config.data.sampling,
            "foreground_fraction_threshold": threshold,
            "small_images": int((weights > 1).sum()),
            "total_images": len(train_set),
            "oversample_factor": config.data.small_lesion_factor,
        }
        (run_dir / "sampling.json").write_text(
            json.dumps(sampling_summary, indent=2), encoding="utf-8"
        )
    train_loader = make_loader(train_set, config, True)
    val_loader = make_loader(val_set, config, False)
    model = DINOv3Segmenter(
        config.model.backbone,
        config.data.num_classes,
        config.model.decoder,
        config.model.freeze_backbone,
        config.model.feature_layers,
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    if config.training.scheduler == "constant":
        scheduler = None
    elif config.training.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs,
            eta_min=config.training.minimum_learning_rate,
        )
    else:
        raise ValueError("training.scheduler must be 'constant' or 'cosine'")
    criterion = nn.CrossEntropyLoss(ignore_index=config.data.ignore_index)
    best_metric = -1.0
    metrics_path = run_dir / "metrics.jsonl"
    epochs = 1 if args.smoke_test else config.training.epochs
    for epoch in range(1, epochs + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, config.training.amp
        )
        validation = evaluate(model, val_loader, criterion, device, config)
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "validation": validation,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + os.linesep)
        print(json.dumps(record, ensure_ascii=False))
        selection_metric = config.training.selection_metric
        if selection_metric not in validation:
            raise KeyError(f"Unknown checkpoint selection metric: {selection_metric}")
        if validation[selection_metric] > best_metric:
            best_metric = float(validation[selection_metric])
            trainable_state = {
                name: tensor
                for name, tensor in model.state_dict().items()
                if name.startswith("decoder.")
            }
            torch.save(
                {"epoch": epoch, "model": trainable_state, "config": config.as_dict()},
                run_dir / "best.pt",
            )
        if scheduler is not None:
            scheduler.step()


if __name__ == "__main__":
    main()
