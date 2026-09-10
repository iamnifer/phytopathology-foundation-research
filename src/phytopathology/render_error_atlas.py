from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch

from .config import load_config
from .data import PlantSegDataset
from .model import build_model, load_trainable_state
from .processing import build_processor
from .visualize_errors import denormalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a contact-sheet atlas of hard cases")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--per-image", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--per-page", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def error_overlay(
    image: np.ndarray, prediction: torch.Tensor, target: torch.Tensor, alpha: float = 0.62
) -> np.ndarray:
    result = image.copy()
    predicted = prediction.numpy() == 1
    actual = target.numpy() == 1
    categories = (
        (predicted & actual, np.array([0.10, 0.85, 0.10])),
        (predicted & ~actual, np.array([1.00, 0.10, 0.10])),
        (~predicted & actual, np.array([0.10, 0.35, 1.00])),
    )
    for mask, color in categories:
        result[mask] = (1 - alpha) * result[mask] + alpha * color
    return result


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.count < 1 or args.per_page < 1:
        raise ValueError("count and per-page must be positive")
    config = load_config(args.config)
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
    index_by_name = {str(name): index for index, (name, _) in enumerate(dataset.samples)}
    frame = pd.read_csv(args.per_image).sort_values("foreground_iou").head(args.count)
    unknown = set(frame["image"].astype(str)).difference(index_by_name)
    if unknown:
        raise ValueError(f"Unknown images in per-image table: {sorted(unknown)[:5]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config.model, config.data.num_classes).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    load_trainable_state(model, checkpoint["model"])
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    pages = math.ceil(len(frame) / args.per_page)
    for page_index in range(pages):
        page = frame.iloc[page_index * args.per_page : (page_index + 1) * args.per_page]
        columns = 2
        rows = math.ceil(len(page) / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(11.7, 3.4 * rows), squeeze=False)
        for axis, (_, record) in zip(axes.flat, page.iterrows(), strict=False):
            name = str(record["image"])
            dataset_index = index_by_name[name]
            image, target = dataset[dataset_index]
            prediction = (
                model(image.unsqueeze(0).to(device)).argmax(dim=1).squeeze(0).cpu()
            )
            overlay = error_overlay(denormalize(image, processor), prediction, target)
            axis.imshow(overlay)
            axis.set_title(
                f"#{len(manifest) + 1} {Path(name).name}\n"
                f"IoU={record['foreground_iou']:.3f}; "
                f"эталон={record['gt_foreground_fraction']:.1%}; "
                f"прогноз={record['pred_foreground_fraction']:.1%}",
                fontsize=9,
            )
            axis.axis("off")
            manifest.append(
                {
                    "rank": len(manifest) + 1,
                    "image": name,
                    "dataset_index": dataset_index,
                    "foreground_iou": float(record["foreground_iou"]),
                    "foreground_precision": float(record["foreground_precision"]),
                    "foreground_recall": float(record["foreground_recall"]),
                    "gt_foreground_fraction": float(record["gt_foreground_fraction"]),
                    "pred_foreground_fraction": float(record["pred_foreground_fraction"]),
                }
            )
        for axis in axes.flat[len(page) :]:
            axis.axis("off")
        figure.legend(
            handles=(
                Patch(color=(0.10, 0.85, 0.10), label="Верно"),
                Patch(color=(1.00, 0.10, 0.10), label="Ложное срабатывание"),
                Patch(color=(0.10, 0.35, 1.00), label="Пропуск"),
            ),
            loc="lower center",
            ncol=3,
        )
        figure.tight_layout(rect=(0, 0.03, 1, 1))
        figure.savefig(
            args.output_dir / f"worst_cases_{page_index + 1:02d}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(figure)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
