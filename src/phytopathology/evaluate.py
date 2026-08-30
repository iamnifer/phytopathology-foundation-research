from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .config import load_config
from .data import PlantSegDataset
from .model import build_model, load_trainable_state
from .processing import build_processor
from .train import evaluate, make_loader, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved PlantSeg model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
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
        binary_masks=config.data.binary_masks,
        resize_mode=config.data.resize_mode,
    )
    model = build_model(config.model, config.data.num_classes).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    load_trainable_state(model, checkpoint["model"])

    criterion = nn.CrossEntropyLoss(ignore_index=config.data.ignore_index)
    result = evaluate(model, make_loader(dataset, config, False), criterion, device, config)
    result.update({"split": args.split, "checkpoint": str(args.checkpoint)})
    output = args.output or args.checkpoint.parent / f"{args.split}_metrics.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
