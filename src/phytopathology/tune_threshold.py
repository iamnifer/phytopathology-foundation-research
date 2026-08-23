from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoImageProcessor

from .config import load_config
from .data import PlantSegDataset
from .metrics import BinaryThresholdSweep
from .model import DINOv3Segmenter
from .train import autocast_context, make_loader, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune a binary foreground threshold on the validation split"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum", type=float, default=0.05)
    parser.add_argument("--maximum", type=float, default=0.95)
    parser.add_argument("--steps", type=int, default=91)
    parser.add_argument(
        "--selection-metric", choices=("foreground_iou", "miou"), default="foreground_iou"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if not config.data.binary_masks or config.data.num_classes != 2:
        raise ValueError("Threshold tuning requires a two-class binary config")
    if args.steps < 2 or not 0 <= args.minimum < args.maximum <= 1:
        raise ValueError("Expected at least two thresholds with 0 <= min < max <= 1")

    set_seed(config.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(config.model.backbone)
    dataset = PlantSegDataset(
        config.data.root,
        "val",
        processor,
        config.data.image_size,
        config.data.num_classes,
        config.data.ignore_index,
        config.data.metadata_file,
        binary_masks=True,
        resize_mode=config.data.resize_mode,
    )
    model = DINOv3Segmenter(
        config.model.backbone,
        config.data.num_classes,
        config.model.decoder,
        config.model.freeze_backbone,
        config.model.feature_layers,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if unexpected or any(not name.startswith("backbone.") for name in missing):
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    sweep = BinaryThresholdSweep(ignore_index=config.data.ignore_index)
    model.eval()
    with torch.inference_mode():
        for images, targets in tqdm(make_loader(dataset, config, False), desc="val threshold"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast_context(device, config.training.amp):
                probability = model(images).softmax(dim=1)[:, 1]
            sweep.update(probability, targets)

    thresholds = torch.linspace(args.minimum, args.maximum, args.steps).tolist()
    curve = sweep.compute(thresholds)
    best = max(curve, key=lambda row: row[args.selection_metric])
    result = {
        "split": "val",
        "checkpoint": str(args.checkpoint),
        "selection_metric": args.selection_metric,
        "best": best,
        "argmax_equivalent": min(curve, key=lambda row: abs(row["threshold"] - 0.5)),
        "curve": curve,
    }
    output = args.output or args.checkpoint.parent / "val_threshold_sweep.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "curve"}))


if __name__ == "__main__":
    main()
