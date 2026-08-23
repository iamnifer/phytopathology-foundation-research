from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

from .metrics import BinaryAveragePrecision, SegmentationMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate oracle-prompted SAM on PlantSeg")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", choices=("point", "box"), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def positive_point(mask: np.ndarray) -> list[float]:
    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        raise ValueError("positive point requires a non-empty foreground mask")
    center = np.array([rows.mean(), columns.mean()])
    nearest = np.argmin((rows - center[0]) ** 2 + (columns - center[1]) ** 2)
    return [float(columns[nearest]), float(rows[nearest])]


def bounding_box(mask: np.ndarray) -> list[float]:
    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        raise ValueError("bounding box requires a non-empty foreground mask")
    return [float(columns.min()), float(rows.min()), float(columns.max()), float(rows.max())]


def load_validation(data_root: Path, limit: int | None) -> list[tuple[Path, Path]]:
    frame = pd.read_csv(data_root / "Metadatav2.csv")
    frame = frame[frame["Split"] == "Validation"]
    if limit is not None:
        frame = frame.iloc[:limit]
    image_dir = data_root / "images" / "val"
    mask_dir = data_root / "annotations" / "val"
    if not image_dir.is_dir():
        image_dir = data_root / "images" / "validation"
        mask_dir = data_root / "annotations" / "validation"
    return [
        (image_dir / str(row["Name"]), mask_dir / str(row["Label file"]))
        for _, row in frame.iterrows()
    ]


def batches(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = SamProcessor.from_pretrained(args.model)
    model = SamModel.from_pretrained(args.model).to(device).eval()
    samples = load_validation(args.data_root, args.limit)
    metrics = SegmentationMetrics(2, ignore_index=255)
    average_precision = BinaryAveragePrecision(ignore_index=255)
    per_image = []

    for group in batches(samples, args.batch_size):
        images = []
        targets = []
        prompts = []
        for image_path, mask_path in group:
            with Image.open(image_path) as source:
                images.append(source.convert("RGB"))
            with Image.open(mask_path) as source:
                raw = np.asarray(source.convert("L"))
            target = np.where(raw == 255, 255, raw > 0).astype(np.int64)
            targets.append(torch.from_numpy(target))
            foreground = (raw > 0) & (raw != 255)
            prompt = (
                positive_point(foreground)
                if args.prompt == "point"
                else bounding_box(foreground)
            )
            prompts.append(prompt)

        prompt_inputs = (
            {
                "input_points": [[[point]] for point in prompts],
                "input_labels": [[[1]] for _ in prompts],
            }
            if args.prompt == "point"
            else {"input_boxes": [[box] for box in prompts]}
        )
        inputs = processor(images=images, return_tensors="pt", **prompt_inputs)
        model_inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        outputs = model(**model_inputs, multimask_output=True)
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
            binarize=False,
        )
        choices = outputs.iou_scores[:, 0].argmax(dim=-1).cpu()

        for index, ((image_path, _), target, candidate_masks, choice) in enumerate(
            zip(group, targets, masks, choices, strict=True)
        ):
            logits = candidate_masks[0, int(choice)]
            prediction = (logits > 0).to(torch.int64)
            metrics.update(prediction, target)
            average_precision.update(logits.sigmoid(), target)
            valid = target != 255
            truth = (target == 1) & valid
            predicted = (prediction == 1) & valid
            union = (truth | predicted).sum().item()
            image_iou = float((truth & predicted).sum().item() / union) if union else 1.0
            per_image.append(
                {
                    "name": image_path.name,
                    "iou": image_iou,
                    "sam_iou_score": float(outputs.iou_scores[index, 0, choice].item()),
                }
            )

    summary = metrics.compute()
    summary["pixel_average_precision"] = average_precision.compute()
    summary["mean_per_image_iou"] = float(np.mean([row["iou"] for row in per_image]))
    summary["prompt"] = args.prompt
    summary["samples"] = len(samples)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output / "per_image.jsonl").open("w", encoding="utf-8") as stream:
        for row in per_image:
            stream.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
