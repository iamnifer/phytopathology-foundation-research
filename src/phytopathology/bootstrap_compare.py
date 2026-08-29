from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COUNT_COLUMNS = ("true_positive", "false_positive", "false_negative")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired image bootstrap for foreground IoU")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def dataset_iou(counts: np.ndarray) -> np.ndarray:
    denominator = counts[..., 0] + counts[..., 1] + counts[..., 2]
    return np.divide(
        counts[..., 0], denominator, out=np.full_like(denominator, np.nan), where=denominator > 0
    )


def paired_bootstrap(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    iterations: int = 10_000,
    seed: int = 2026,
) -> dict[str, float | int]:
    required = {"image", *COUNT_COLUMNS}
    for name, frame in (("baseline", baseline), ("candidate", candidate)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        if frame["image"].duplicated().any():
            raise ValueError(f"{name} contains duplicate image identifiers")
    left = baseline.set_index("image").sort_index()
    right = candidate.set_index("image").sort_index()
    if not left.index.equals(right.index):
        raise ValueError("baseline and candidate image sets differ")

    base_counts = left.loc[:, COUNT_COLUMNS].to_numpy(dtype=np.float64)
    candidate_counts = right.loc[:, COUNT_COLUMNS].to_numpy(dtype=np.float64)
    point_difference = float(
        dataset_iou(candidate_counts.sum(axis=0)) - dataset_iou(base_counts.sum(axis=0))
    )
    rng = np.random.default_rng(seed)
    differences = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 250):
        width = min(250, iterations - start)
        indices = rng.integers(0, len(left), size=(width, len(left)))
        sampled_base = base_counts[indices].sum(axis=1)
        sampled_candidate = candidate_counts[indices].sum(axis=1)
        differences[start : start + width] = dataset_iou(sampled_candidate) - dataset_iou(
            sampled_base
        )
    lower, upper = np.quantile(differences, [0.025, 0.975])
    p_two_sided = min(1.0, 2 * min(np.mean(differences <= 0), np.mean(differences >= 0)))
    return {
        "images": len(left),
        "iterations": iterations,
        "baseline_foreground_iou": float(dataset_iou(base_counts.sum(axis=0))),
        "candidate_foreground_iou": float(dataset_iou(candidate_counts.sum(axis=0))),
        "difference": point_difference,
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "p_two_sided": float(p_two_sided),
        "seed": seed,
    }


def main() -> None:
    args = parse_args()
    result = paired_bootstrap(
        pd.read_csv(args.baseline),
        pd.read_csv(args.candidate),
        iterations=args.iterations,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
