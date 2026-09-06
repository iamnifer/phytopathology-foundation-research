from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


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


def kmeans_representatives(
    descriptors: np.ndarray,
    count: int,
    seed: int,
    iterations: int = 30,
    restarts: int = 3,
) -> np.ndarray:
    if not 0 < count <= len(descriptors):
        raise ValueError("count must lie between one and the pool size")
    if iterations < 1 or restarts < 1:
        raise ValueError("iterations and restarts must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = F.normalize(torch.from_numpy(descriptors).float().to(device), dim=1)
    rng = np.random.default_rng(seed)
    best_objective = -float("inf")
    best_centers = None
    best_labels = None
    for _ in range(restarts):
        initial = rng.choice(len(features), size=count, replace=False)
        centers = features[torch.from_numpy(initial).to(device)].clone()
        for _ in range(iterations):
            similarities = features @ centers.T
            maximum_similarity, labels = similarities.max(dim=1)
            sums = torch.zeros_like(centers)
            sums.index_add_(0, labels, features)
            counts = torch.bincount(labels, minlength=count)
            present = counts > 0
            centers[present] = sums[present] / counts[present, None]
            missing = (~present).sum().item()
            if missing:
                replacements = maximum_similarity.argsort()[:missing]
                centers[~present] = features[replacements]
            centers = F.normalize(centers, dim=1)
        similarities = features @ centers.T
        maximum_similarity, labels = similarities.max(dim=1)
        objective = float(maximum_similarity.mean())
        if objective > best_objective:
            best_objective = objective
            best_centers = centers.cpu().numpy()
            best_labels = labels.cpu().numpy()

    if best_centers is None or best_labels is None:
        raise RuntimeError("spherical k-means failed to produce a solution")
    selected = np.full(count, -1, dtype=np.int64)
    empty_clusters = []
    for cluster in range(count):
        members = np.flatnonzero(best_labels == cluster)
        if len(members) == 0:
            empty_clusters.append(cluster)
            continue
        similarities = descriptors[members] @ best_centers[cluster]
        selected[cluster] = members[similarities.argmax()]
    available = np.ones(len(descriptors), dtype=bool)
    available[selected[selected >= 0]] = False
    for cluster in empty_clusters:
        candidates = np.flatnonzero(available)
        similarities = descriptors[candidates] @ best_centers[cluster]
        choice = candidates[similarities.argmax()]
        selected[cluster] = choice
        available[choice] = False
    return selected


def main() -> None:
    args = parse_args()
    archive = np.load(args.features)
    descriptors = archive["descriptors"]
    names = archive["names"].astype(str)
    metadata = pd.read_csv(args.metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    order_methods = {
        "random": lambda: random_order(len(names), args.seed),
        "stratified": lambda: stratified_order(names, metadata, args.seed),
        "farthest": lambda: farthest_first_order(descriptors),
    }
    for method in args.methods:
        if method not in {*order_methods, "kmeans"}:
            raise ValueError(f"unknown selection method: {method}")
        order_started = time.perf_counter()
        order = order_methods[method]() if method in order_methods else None
        order_seconds = time.perf_counter() - order_started
        for budget in args.budgets:
            if not 0 < budget < 1:
                raise ValueError("budgets must lie strictly between 0 and 1")
            count = max(1, math.ceil(len(names) * budget))
            if method == "kmeans":
                selection_started = time.perf_counter()
                selected = kmeans_representatives(descriptors, count, args.seed)
                selection_seconds = time.perf_counter() - selection_started
            else:
                selected = order[:count]
                selection_seconds = order_seconds
            manifest = {
                "method": method,
                "seed": args.seed if method in {"random", "stratified", "kmeans"} else None,
                "budget_fraction": budget,
                "pool_size": len(names),
                "selection_seconds": selection_seconds,
                "samples": names[selected].tolist(),
            }
            if method == "kmeans":
                manifest["selection_config"] = {
                    "algorithm": "spherical_kmeans",
                    "distance": "cosine",
                    "iterations": 30,
                    "restarts": 3,
                }
            path = args.output_dir / f"{method}_seed{args.seed}_{budget:g}.json"
            path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"{path}: {count} samples")


if __name__ == "__main__":
    main()
