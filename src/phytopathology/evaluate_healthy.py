from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import load_config
from .model import build_model, load_trainable_state
from .processing import build_processor
from .train import autocast_context, set_seed


class HealthyImageDataset(Dataset[tuple[torch.Tensor, str, str]]):
    def __init__(self, root: Path, records: list[dict[str, str]], processor, size: int) -> None:
        self.root = root
        self.records = records
        self.processor = processor
        self.size = size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, str]:
        record = self.records[index]
        with Image.open(self.root / record["path"]) as source:
            image = source.convert("RGB").resize(
                (self.size, self.size), Image.Resampling.BILINEAR
            )
        pixels = self.processor(images=image, return_tensors="pt", do_resize=False)
        return pixels.pixel_values.squeeze(0), record["path"], record["label"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate false alarms on healthy images")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def summarize(fractions: np.ndarray) -> dict[str, float | int]:
    result: dict[str, float | int] = {
        "images": len(fractions),
        "mean_predicted_lesion_fraction": float(fractions.mean()),
        "median_predicted_lesion_fraction": float(np.median(fractions)),
        "q90_predicted_lesion_fraction": float(np.quantile(fractions, 0.9)),
        "q95_predicted_lesion_fraction": float(np.quantile(fractions, 0.95)),
    }
    for threshold in (0.001, 0.005, 0.01, 0.05):
        key = f"images_above_{threshold:g}_lesion_fraction"
        result[key] = float((fractions > threshold).mean())
    return result


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if not config.data.binary_masks or config.data.num_classes != 2:
        raise ValueError("Healthy evaluation requires a binary two-class config")
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = [record for record in records if record.get("split") == args.split]
    missing = [
        record["path"]
        for record in records
        if not (args.data_root / record["path"]).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first: {missing[0]}")

    set_seed(config.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = build_processor(config.model)
    dataset = HealthyImageDataset(args.data_root, records, processor, config.data.image_size)
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_model(config.model, config.data.num_classes).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    load_trainable_state(model, checkpoint["model"])
    model.eval()

    rows = []
    for images, paths, labels in tqdm(loader, desc="healthy evaluation"):
        with autocast_context(device, config.training.amp):
            logits = model(images.to(device, non_blocking=True))
        probability = logits.softmax(dim=1)[:, 1]
        prediction = logits.argmax(dim=1) == 1
        fractions = prediction.float().mean(dim=(1, 2)).cpu().numpy()
        mean_probabilities = probability.mean(dim=(1, 2)).float().cpu().numpy()
        rows.extend(
            {
                "image": path,
                "healthy_class": label,
                "predicted_lesion_fraction": float(fraction),
                "mean_lesion_probability": float(mean_probability),
            }
            for path, label, fraction, mean_probability in zip(
                paths, labels, fractions, mean_probabilities, strict=True
            )
        )
    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "per_image.csv", index=False)
    summary = summarize(frame["predicted_lesion_fraction"].to_numpy())
    summary.update(
        {
            "split": args.split,
            "source": "PlantWild v1 healthy classes",
            "checkpoint": str(args.checkpoint),
            "image_size": config.data.image_size,
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
