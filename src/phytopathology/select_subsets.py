from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select nested PlantSeg training subsets")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["random", "stratified", "farthest"])
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.25, 0.5])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def random_order(size: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(size)


def stratified_order(names: np.ndarray, metadata: pd.DataFrame, seed: int) -> np.ndarray:
    diseases = metadata.set_index("Name").loc[names, "Disease"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    priorities = np.empty(len(names), dtype=np.float64)
    for disease in np.unique(diseases):
        indices = np.flatnonzero(diseases == disease)
        shuffled = rng.permutation(indices)
        priorities[shuffled] = (np.arange(len(indices)) + rng.random()) / len(indices)
    return np.argsort(priorities, kind="stable")


def farthest_first_order(descriptors: np.ndarray) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = torch.from_numpy(descriptors).to(device)
    center = features.mean(dim=0, keepdim=True)
    first = int(torch.linalg.vector_norm(features - center, dim=1).argmax())
    order = torch.empty(len(features), dtype=torch.long, device=device)
    selected = torch.zeros(len(features), dtype=torch.bool, device=device)
    distances = torch.full((len(features),), torch.inf, device=device)
    current = first
    for position in range(len(features)):
        order[position] = current
        selected[current] = True
        distance = 2.0 - 2.0 * (features @ features[current])
        distances = torch.minimum(distances, distance)
        distances[selected] = -1
        current = int(distances.argmax())
    return order.cpu().numpy()


def main() -> None:
    args = parse_args()
    archive = np.load(args.features)
    descriptors = archive["descriptors"]
    names = archive["names"].astype(str)
    metadata = pd.read_csv(args.metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = {
        "random": lambda: random_order(len(names), args.seed),
        "stratified": lambda: stratified_order(names, metadata, args.seed),
        "farthest": lambda: farthest_first_order(descriptors),
    }
    for method in args.methods:
        if method not in methods:
            raise ValueError(f"unknown selection method: {method}")
        order = methods[method]()
        for budget in args.budgets:
            if not 0 < budget < 1:
                raise ValueError("budgets must lie strictly between 0 and 1")
            count = max(1, math.ceil(len(names) * budget))
            manifest = {
                "method": method,
                "seed": args.seed if method != "farthest" else None,
                "budget_fraction": budget,
                "pool_size": len(names),
                "samples": names[order[:count]].tolist(),
            }
            path = args.output_dir / f"{method}_seed{args.seed}_{budget:g}.json"
            path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"{path}: {count} samples")


if __name__ == "__main__":
    main()
