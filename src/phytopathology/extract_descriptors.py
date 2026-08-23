from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from .config import load_config
from .data import PlantSegDataset
from .train import autocast_context, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen DINOv3 train descriptors")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(config.model.backbone)
    dataset = PlantSegDataset(
        config.data.root,
        "train",
        processor,
        config.data.image_size,
        config.data.num_classes,
        config.data.ignore_index,
        config.data.metadata_file,
        binary_masks=config.data.binary_masks,
        resize_mode=config.data.resize_mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.data.num_workers > 0,
    )
    model = AutoModel.from_pretrained(config.model.backbone).to(device).eval()
    descriptors = []
    for images, _ in tqdm(loader, desc="DINOv3 descriptors"):
        images = images.to(device, non_blocking=True)
        with autocast_context(device, config.training.amp):
            tokens = model(pixel_values=images).last_hidden_state
        register_tokens = int(getattr(model.config, "num_register_tokens", 0))
        pooled = tokens[:, 1 + register_tokens :].float().mean(dim=1)
        pooled = torch.nn.functional.normalize(pooled, dim=1)
        descriptors.append(pooled.cpu().numpy())

    output = np.concatenate(descriptors).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        descriptors=output,
        names=np.asarray([str(name) for name, _ in dataset.samples]),
    )
    metadata = {
        "samples": len(dataset),
        "dimensions": output.shape[1],
        "backbone": config.model.backbone,
        "pooling": "mean_patch_tokens_l2_normalized",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
