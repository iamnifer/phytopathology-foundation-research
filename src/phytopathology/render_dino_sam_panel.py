from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle
from PIL import Image
from torch.nn import functional as F
from transformers import SamModel, SamProcessor

from .config import load_config
from .data import PlantSegDataset
from .model import build_model, load_trainable_state
from .processing import build_processor
from .sam_evaluate import bounding_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render comparable DINOv3 and oracle-box SAM cases"
    )
    parser.add_argument("--dino-config", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--sam-model", type=Path, required=True)
    parser.add_argument("--per-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def select_cases(frame: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    ordered = frame.sort_values("foreground_iou").reset_index(drop=True)
    candidates = [
        ("Худший", ordered.iloc[0]),
        ("Медианный", ordered.iloc[len(ordered) // 2]),
        ("Лучший", ordered.iloc[-1]),
    ]
    small = ordered[
        ordered["gt_foreground_fraction"]
        <= ordered["gt_foreground_fraction"].quantile(0.25)
    ].copy()
    small_median = small["foreground_iou"].median()
    small["distance_to_median"] = (small["foreground_iou"] - small_median).abs()
    already_selected = {str(record["image"]) for _, record in candidates}
    small = small[~small["image"].astype(str).isin(already_selected)]
    candidates.append(
        ("Малое поражение", small.sort_values("distance_to_median").iloc[0])
    )
    return candidates


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    result = image.astype(np.float32) / 255.0
    result[mask] = 0.42 * result[mask] + 0.58 * np.asarray(color)
    return result


def mask_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    return float(intersection / union) if union else 1.0


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    config = load_config(args.dino_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_processor = build_processor(config.model)
    dataset = PlantSegDataset(
        config.data.root,
        "val",
        dino_processor,
        config.data.image_size,
        config.data.num_classes,
        config.data.ignore_index,
        config.data.metadata_file,
        binary_masks=True,
        resize_mode=config.data.resize_mode,
    )
    index_by_name = {str(name): index for index, (name, _) in enumerate(dataset.samples)}
    cases = select_cases(pd.read_csv(args.per_image))

    dino = build_model(config.model, config.data.num_classes).to(device)
    checkpoint = torch.load(args.dino_checkpoint, map_location="cpu", weights_only=True)
    load_trainable_state(dino, checkpoint["model"])
    dino.eval()
    sam_processor = SamProcessor.from_pretrained(args.sam_model)
    sam = SamModel.from_pretrained(args.sam_model).to(device).eval()

    figure, axes = plt.subplots(len(cases), 4, figsize=(13.5, 3.4 * len(cases)))
    manifest = []
    for row_index, (case_name, record) in enumerate(cases):
        name = str(record["image"])
        dataset_index = index_by_name[name]
        image_path = dataset.image_dir / name
        _, mask_name = dataset.samples[dataset_index]
        with Image.open(image_path) as source:
            pil_image = source.convert("RGB")
        image = np.asarray(pil_image)
        with Image.open(dataset.mask_dir / str(mask_name)) as source:
            raw = np.asarray(source.convert("L"))
        target = (raw > 0) & (raw != config.data.ignore_index)

        dino_image, _ = dataset[dataset_index]
        dino_logits = dino(dino_image.unsqueeze(0).to(device)).float()
        dino_logits = F.interpolate(
            dino_logits, size=target.shape, mode="bilinear", align_corners=False
        )
        dino_prediction = dino_logits.argmax(dim=1).squeeze(0).cpu().numpy() == 1

        box = bounding_box(target)
        sam_inputs = sam_processor(
            images=[pil_image], input_boxes=[[box]], return_tensors="pt"
        )
        sam_model_inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in sam_inputs.items()
        }
        sam_outputs = sam(**sam_model_inputs, multimask_output=True)
        sam_masks = sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"],
            sam_inputs["reshaped_input_sizes"],
            binarize=False,
        )[0]
        choice = int(sam_outputs.iou_scores[0, 0].argmax().cpu())
        sam_prediction = sam_masks[0, choice].numpy() > 0

        dino_iou = mask_iou(dino_prediction, target)
        sam_iou = mask_iou(sam_prediction, target)
        panels = (
            (image, f"{case_name}: изображение"),
            (overlay(image, target, (0.15, 0.85, 0.15)), "Эталон"),
            (overlay(image, dino_prediction, (1.00, 0.25, 0.10)), f"DINOv3, IoU={dino_iou:.3f}"),
            (overlay(image, sam_prediction, (0.15, 0.45, 1.00)), f"SAM, IoU={sam_iou:.3f}"),
        )
        for column_index, (panel, title) in enumerate(panels):
            axis = axes[row_index, column_index]
            axis.imshow(panel)
            axis.set_title(title, fontsize=10)
            axis.axis("off")
            if column_index == 3:
                x_min, y_min, x_max, y_max = box
                axis.add_patch(
                    Rectangle(
                        (x_min, y_min),
                        x_max - x_min,
                        y_max - y_min,
                        fill=False,
                        edgecolor="yellow",
                        linewidth=2,
                    )
                )
        manifest.append(
            {
                "case": case_name,
                "image": name,
                "gt_foreground_fraction": float(target.mean()),
                "dino_iou_native_grid": dino_iou,
                "sam_box_iou_native_grid": sam_iou,
                "oracle_box": box,
            }
        )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
