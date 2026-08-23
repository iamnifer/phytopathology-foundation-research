from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor

from .config import load_config
from .data import PlantSegDataset
from .model import DINOv3Segmenter
from .train import autocast_context, make_loader, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze binary segmentation errors per image")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    result = numerator.to(torch.float64) / denominator.clamp_min(1)
    return torch.where(denominator > 0, result, torch.nan)


@torch.inference_mode()
def collect_rows(
    model, loader, device, amp: bool, ignore_index: int, metadata: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    offset = 0
    model.eval()
    for images, targets in tqdm(loader, desc="per-image analysis"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(device, amp):
            predictions = model(images).argmax(dim=1)
        valid = targets != ignore_index
        predicted = (predictions == 1) & valid
        actual = (targets == 1) & valid
        true_positive = (predicted & actual).sum(dim=(1, 2)).cpu()
        false_positive = (predicted & ~actual & valid).sum(dim=(1, 2)).cpu()
        false_negative = (~predicted & actual).sum(dim=(1, 2)).cpu()
        valid_count = valid.sum(dim=(1, 2)).cpu()
        actual_count = actual.sum(dim=(1, 2)).cpu()
        predicted_count = predicted.sum(dim=(1, 2)).cpu()
        iou = safe_ratio(true_positive, true_positive + false_positive + false_negative)
        precision = safe_ratio(true_positive, true_positive + false_positive)
        recall = safe_ratio(true_positive, true_positive + false_negative)
        for index in range(images.shape[0]):
            meta = metadata.iloc[offset + index]
            rows.append(
                {
                    "image": meta["Name"],
                    "plant": meta["Plant"],
                    "disease": meta["Disease"],
                    "foreground_iou": float(iou[index]),
                    "foreground_precision": float(precision[index]),
                    "foreground_recall": float(recall[index]),
                    "gt_foreground_fraction": float(actual_count[index] / valid_count[index]),
                    "pred_foreground_fraction": float(predicted_count[index] / valid_count[index]),
                }
            )
        offset += images.shape[0]
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    ratio = frame["pred_foreground_fraction"] / frame["gt_foreground_fraction"].clip(1e-12)
    size_group = pd.qcut(
        frame["gt_foreground_fraction"], 4, labels=("q1_small", "q2", "q3", "q4_large")
    )
    by_size = (
        frame.assign(lesion_size_quartile=size_group)
        .groupby("lesion_size_quartile", observed=True)["foreground_iou"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .to_dict(orient="records")
    )
    by_disease = (
        frame.groupby(["plant", "disease"])["foreground_iou"]
        .agg(["count", "mean", "median"])
        .query("count >= 5")
        .sort_values("mean")
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "images": len(frame),
        "macro_foreground_iou_mean": frame["foreground_iou"].mean(),
        "macro_foreground_iou_median": frame["foreground_iou"].median(),
        "zero_iou_fraction": (frame["foreground_iou"] == 0).mean(),
        "mean_gt_foreground_fraction": frame["gt_foreground_fraction"].mean(),
        "mean_pred_foreground_fraction": frame["pred_foreground_fraction"].mean(),
        "oversegmentation_fraction_pred_gt_ratio_above_1_25": (ratio > 1.25).mean(),
        "undersegmentation_fraction_pred_gt_ratio_below_0_75": (ratio < 0.75).mean(),
        "by_lesion_size_quartile": by_size,
        "by_disease_at_least_5_images": by_disease,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if not config.data.binary_masks or config.data.num_classes != 2:
        raise ValueError("Per-image analysis requires a two-class binary config")
    set_seed(config.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(config.model.backbone)
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
    if len(metadata) != len(dataset):
        raise RuntimeError("Metadata order does not match the dataset")

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

    frame = collect_rows(
        model,
        make_loader(dataset, config, False),
        device,
        config.training.amp,
        config.data.ignore_index,
        metadata,
    )
    output_dir = args.output_dir or args.checkpoint.parent / f"analysis_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "per_image.csv", index=False)
    summary = summarize(frame)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if not isinstance(value, list)}))


if __name__ == "__main__":
    main()
