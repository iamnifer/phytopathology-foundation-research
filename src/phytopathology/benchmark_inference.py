from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .config import load_config
from .data import PlantSegDataset
from .model import build_model, load_trainable_state
from .processing import build_processor
from .train import autocast_context, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark model-only segmentation inference")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.warmup < 0 or args.repeats < 1:
        raise ValueError("batch-size and repeats must be positive; warmup must be non-negative")
    config = load_config(args.config)
    set_seed(config.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = build_processor(config.model)
    dataset = PlantSegDataset(
        config.data.root,
        "val",
        processor,
        config.data.image_size,
        config.data.num_classes,
        config.data.ignore_index,
        config.data.metadata_file,
        binary_masks=config.data.binary_masks,
        resize_mode=config.data.resize_mode,
        limit=args.batch_size,
    )
    images = torch.stack([dataset[index][0] for index in range(args.batch_size)]).to(device)
    model = build_model(config.model, config.data.num_classes).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    load_trainable_state(model, checkpoint["model"])
    model.eval()

    for _ in range(args.warmup):
        with autocast_context(device, config.training.amp):
            model(images)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(args.repeats):
        with autocast_context(device, config.training.amp):
            model(images)
    synchronize(device)
    elapsed = time.perf_counter() - started

    result = {
        "device": str(device),
        "input_size": config.data.image_size,
        "batch_size": args.batch_size,
        "warmup_batches": args.warmup,
        "measured_batches": args.repeats,
        "mean_batch_milliseconds": 1000 * elapsed / args.repeats,
        "images_per_second": args.batch_size * args.repeats / elapsed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "checkpoint": str(args.checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
