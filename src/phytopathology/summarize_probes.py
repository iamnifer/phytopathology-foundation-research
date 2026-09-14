from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import yaml

EXPERIMENTS = {
    "resnet_matched_24": "resnet50_binary_linear_probe_cosine",
    "dino_matched_24": "dinov3_vitb16_binary_linear_probe_cosine",
    "resnet_native_48": "resnet50_binary_linear_probe_native48_cosine",
}
METRICS = (
    "foreground_iou",
    "miou",
    "foreground_dice",
    "foreground_precision",
    "foreground_recall",
    "pixel_average_precision",
)


def completed_runs(root: Path, experiment_name: str, epochs: int = 30) -> dict[int, Path]:
    selected: dict[int, Path] = {}
    for run in sorted(root.glob(f"*_{experiment_name}")):
        config_path = run / "config.yaml"
        metrics_path = run / "metrics.jsonl"
        if not config_path.is_file() or not metrics_path.is_file():
            continue
        records = metrics_path.read_text(encoding="utf-8").splitlines()
        if len(records) != epochs:
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        selected[int(config["experiment"]["seed"])] = run
    return selected


def summarize_group(runs: dict[int, Path]) -> dict[str, object]:
    records = []
    for seed, run in sorted(runs.items()):
        history = [
            json.loads(line)
            for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        best = max(history, key=lambda row: row["validation"]["foreground_iou"])
        model = json.loads((run / "model.json").read_text(encoding="utf-8"))
        runtime = json.loads((run / "runtime.json").read_text(encoding="utf-8"))
        records.append(
            {
                "seed": seed,
                "run": str(run),
                "best_epoch": int(best["epoch"]),
                **{name: float(best["validation"][name]) for name in METRICS},
                "total_parameters": int(model["total"]),
                "trainable_parameters": int(model["trainable"]),
                "runtime_seconds": float(runtime["total_seconds"]),
                "peak_cuda_memory_bytes": runtime["peak_cuda_memory_bytes"],
            }
        )
    aggregate = {}
    for name in (*METRICS, "runtime_seconds"):
        values = [float(record[name]) for record in records]
        aggregate[name] = {
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        }
    return {"runs": records, "aggregate": aggregate}


def summarize(root: Path) -> dict[str, object]:
    groups = {
        label: summarize_group(runs)
        for label, name in EXPERIMENTS.items()
        if (runs := completed_runs(root, name))
    }
    result: dict[str, object] = {"groups": groups}
    resnet = groups.get("resnet_matched_24")
    dino = groups.get("dino_matched_24")
    if resnet and dino:
        resnet_by_seed = {row["seed"]: row for row in resnet["runs"]}
        dino_by_seed = {row["seed"]: row for row in dino["runs"]}
        seeds = sorted(set(resnet_by_seed).intersection(dino_by_seed))
        differences = [
            dino_by_seed[seed]["foreground_iou"] - resnet_by_seed[seed]["foreground_iou"]
            for seed in seeds
        ]
        result["paired_dino_minus_resnet_24"] = {
            "seeds": seeds,
            "differences": differences,
            "mean": statistics.mean(differences),
            "sample_std": statistics.stdev(differences) if len(differences) > 1 else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize frozen linear-probe runs")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.runs_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
