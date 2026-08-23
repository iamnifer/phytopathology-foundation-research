from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor

from .config import load_config
from .data import PlantSegDataset
from .model import DINOv3Segmenter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render best, typical and worst binary predictions"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--num-each", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_model(config, checkpoint_path: Path, device: torch.device) -> DINOv3Segmenter:
    model = DINOv3Segmenter(
        config.model.backbone,
        config.data.num_classes,
        config.model.decoder,
        config.model.freeze_backbone,
        config.model.feature_layers,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if unexpected or any(not name.startswith("backbone.") for name in missing):
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model.eval()


@torch.inference_mode()
def rank_predictions(model, dataset, device: torch.device) -> list[tuple[float, int]]:
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8)
    ranked: list[tuple[float, int]] = []
    offset = 0
    for images, targets in tqdm(loader, desc="ranking"):
        predictions = model(images.to(device)).argmax(dim=1).cpu()
        foreground_prediction = predictions == 1
        foreground_target = targets == 1
        intersection = (foreground_prediction & foreground_target).sum(dim=(1, 2))
        union = (foreground_prediction | foreground_target).sum(dim=(1, 2))
        iou = intersection / union.clamp_min(1)
        ranked.extend((float(score), offset + index) for index, score in enumerate(iou))
        offset += images.shape[0]
    return sorted(ranked)


def denormalize(image: torch.Tensor, processor) -> np.ndarray:
    mean = torch.tensor(processor.image_mean).view(3, 1, 1)
    std = torch.tensor(processor.image_std).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


@torch.inference_mode()
def render_sample(model, dataset, processor, device, index: int, title: str, output: Path) -> None:
    image, target = dataset[index]
    prediction = model(image.unsqueeze(0).to(device)).argmax(dim=1).squeeze(0).cpu()
    target_foreground = target == 1
    prediction_foreground = prediction == 1
    error = np.zeros((*target.shape, 3), dtype=np.float32)
    error[(target_foreground & prediction_foreground).numpy()] = (0.1, 0.8, 0.1)
    error[(~target_foreground & prediction_foreground).numpy()] = (1.0, 0.1, 0.1)
    error[(target_foreground & ~prediction_foreground).numpy()] = (0.1, 0.3, 1.0)

    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = (
        (denormalize(image, processor), "Image"),
        (target_foreground.numpy(), "Ground truth"),
        (prediction_foreground.numpy(), "Prediction"),
        (error, "TP green / FP red / FN blue"),
    )
    for axis, (panel, panel_title) in zip(axes, panels, strict=True):
        axis.imshow(panel)
        axis.set_title(panel_title)
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if not config.data.binary_masks:
        raise SystemExit("Error visualization currently requires a binary_masks config")
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
    model = load_model(config, args.checkpoint, device)
    ranked = rank_predictions(model, dataset, device)
    middle = max(0, (len(ranked) - args.num_each) // 2)
    groups = {
        "worst": ranked[: args.num_each],
        "typical": ranked[middle : middle + args.num_each],
        "best": ranked[-args.num_each :],
    }
    manifest = []
    for group, samples in groups.items():
        for rank, (iou, index) in enumerate(samples, start=1):
            image_name = str(dataset.samples[index][0])
            filename = f"{group}_{rank:02d}_iou_{iou:.4f}.png"
            title = f"{group}: IoU={iou:.4f} — {image_name}"
            render_sample(
                model, dataset, processor, device, index, title, args.output_dir / filename
            )
            manifest.append(
                {"group": group, "rank": rank, "iou": iou, "index": index, "image": image_name}
            )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
