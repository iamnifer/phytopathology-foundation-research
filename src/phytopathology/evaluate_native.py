from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm

from .config import load_config
from .data import PlantSegDataset
from .metrics import BinaryAveragePrecision, SegmentationMetrics
from .model import build_model, load_trainable_state
from .processing import build_processor
from .train import autocast_context, make_loader, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate binary predictions on the original annotation grid"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_binary_target(path: Path, ignore_index: int) -> torch.Tensor:
    with Image.open(path) as source:
        raw = torch.from_numpy(np.array(source.convert("L"), dtype=np.int64, copy=True))
    ignored = raw == ignore_index
    target = (raw > 0).to(torch.int64)
    target[ignored] = ignore_index
    return target


def binary_counts(
    prediction: torch.Tensor, target: torch.Tensor, ignore_index: int
) -> dict[str, int]:
    valid = target != ignore_index
    predicted = (prediction == 1) & valid
    actual = (target == 1) & valid
    return {
        "true_positive": int((predicted & actual).sum()),
        "false_positive": int((predicted & ~actual & valid).sum()),
        "false_negative": int((~predicted & actual).sum()),
        "valid_pixels": int(valid.sum()),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if not config.data.binary_masks or config.data.num_classes != 2:
        raise ValueError("Native-resolution evaluation requires a binary two-class config")
    set_seed(config.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = build_processor(config.model)
    dataset = PlantSegDataset(
        config.data.root,
        args.split,
        processor,
        config.data.image_size,
        config.data.num_classes,
        config.data.ignore_index,
        config.data.metadata_file,
        binary_masks=True,
        resize_mode=config.data.resize_mode,
    )
    metadata = pd.read_csv(Path(config.data.root) / config.data.metadata_file)
    metadata = metadata[metadata["Split"] == dataset.SPLIT_NAMES[args.split]].reset_index(drop=True)
    model = build_model(config.model, config.data.num_classes).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    load_trainable_state(model, checkpoint["model"])
    model.eval()

    metrics = SegmentationMetrics(config.data.num_classes, config.data.ignore_index)
    average_precision = BinaryAveragePrecision(ignore_index=config.data.ignore_index)
    rows: list[dict[str, object]] = []
    offset = 0
    for images, _ in tqdm(make_loader(dataset, config, False), desc="native-grid evaluation"):
        with autocast_context(device, config.training.amp):
            logits = model(images.to(device, non_blocking=True))
        for batch_index in range(logits.shape[0]):
            sample_index = offset + batch_index
            _, mask_name = dataset.samples[sample_index]
            target = load_binary_target(dataset.mask_dir / str(mask_name), config.data.ignore_index)
            native_logits = F.interpolate(
                logits[batch_index : batch_index + 1].float(),
                size=target.shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).cpu()
            probability = native_logits.softmax(dim=0)[1]
            prediction = native_logits.argmax(dim=0)
            metrics.update(prediction, target)
            average_precision.update(probability, target)
            counts = binary_counts(prediction, target, config.data.ignore_index)
            union = counts["true_positive"] + counts["false_positive"] + counts["false_negative"]
            meta = metadata.iloc[sample_index]
            rows.append(
                {
                    "image": meta["Name"],
                    "plant": meta["Plant"],
                    "disease": meta["Disease"],
                    "foreground_iou": counts["true_positive"] / union if union else float("nan"),
                    **counts,
                }
            )
        offset += logits.shape[0]

    summary = metrics.compute(config.training.exclude_background_from_miou)
    summary.update(
        {
            "pixel_average_precision": average_precision.compute(),
            "images": len(rows),
            "model_input_size": config.data.image_size,
            "evaluation_grid": "original_annotation_resolution",
            "split": args.split,
            "checkpoint": str(args.checkpoint),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "per_image.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
