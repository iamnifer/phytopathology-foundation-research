from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import yaml

METRICS = (
    "foreground_iou",
    "miou",
    "foreground_dice",
    "foreground_precision",
    "foreground_recall",
    "pixel_average_precision",
)

RUN_NAMES = {
    "dino_practical": "dinov3_vitb16_binary_multilayer_conv_cosine",
    "cnn_practical": "deeplabv3_resnet50_binary_pretrained",
    "dino_probe": "dinov3_vitb16_binary_linear_probe_cosine",
    "resnet_probe": "resnet50_binary_linear_probe_cosine",
}
SEEDS = (42, 43, 44)


def aggregate(records: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    result = {}
    for metric in METRICS:
        values = [float(record[metric]) for record in records]
        result[metric] = {
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values),
        }
    return result


def find_run(runs_root: Path, experiment_name: str, seed: int) -> Path:
    matches = []
    for path in sorted(runs_root.glob(f"*_{experiment_name}")):
        config_path = path / "config.yaml"
        if not config_path.is_file():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if int(config.get("experiment", {}).get("seed", -1)) == seed:
            matches.append(path)
    if not matches:
        raise FileNotFoundError(
            f"No run found for experiment={experiment_name!r}, seed={seed} in {runs_root}"
        )
    return matches[-1]


def load_group(runs_root: Path, group: str, native: bool = False) -> dict[str, object]:
    records = []
    for seed in SEEDS:
        path = find_run(runs_root, RUN_NAMES[group], seed)
        metrics_path = (
            path / "confirmatory_test_native" / "summary.json"
            if native
            else path / "confirmatory_test_resized_metrics.json"
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        records.append({"seed": seed, "run": str(path), **metrics})
    return {"runs": records, "aggregate": aggregate(records)}


def paired_summary(
    left: dict[str, object], right: dict[str, object], right_minus_left: str
) -> dict[str, object]:
    left_by_seed = {row["seed"]: row for row in left["runs"]}
    right_by_seed = {row["seed"]: row for row in right["runs"]}
    seeds = sorted(set(left_by_seed).intersection(right_by_seed))
    differences = [
        float(right_by_seed[seed]["foreground_iou"]) - float(left_by_seed[seed]["foreground_iou"])
        for seed in seeds
    ]
    return {
        "definition": right_minus_left,
        "seeds": seeds,
        "differences": differences,
        "mean": statistics.mean(differences),
        "sample_std": statistics.stdev(differences),
    }


def summarize(runs_root: Path, bootstrap_root: Path) -> dict[str, object]:
    groups = {
        "dino_practical_resized": load_group(runs_root, "dino_practical"),
        "cnn_practical_resized": load_group(runs_root, "cnn_practical"),
        "dino_practical_native": load_group(runs_root, "dino_practical", native=True),
        "cnn_practical_native": load_group(runs_root, "cnn_practical", native=True),
        "dino_probe_resized": load_group(runs_root, "dino_probe"),
        "resnet_probe_resized": load_group(runs_root, "resnet_probe"),
    }
    comparisons = {
        "practical_resized_cnn_minus_dino": paired_summary(
            groups["dino_practical_resized"],
            groups["cnn_practical_resized"],
            "cnn_minus_dino",
        ),
        "practical_native_cnn_minus_dino": paired_summary(
            groups["dino_practical_native"],
            groups["cnn_practical_native"],
            "cnn_minus_dino",
        ),
        "probe_resized_dino_minus_resnet": paired_summary(
            groups["resnet_probe_resized"],
            groups["dino_probe_resized"],
            "dino_minus_resnet",
        ),
    }
    bootstrap = {
        "practical_native_cnn_minus_dino": {
            str(seed): json.loads(
                (bootstrap_root / f"test_native_cnn_vs_dino_seed{seed}.json").read_text(
                    encoding="utf-8"
                )
            )
            for seed in (42, 43, 44)
        },
        "probe_resized_dino_minus_resnet": {
            str(seed): json.loads(
                (bootstrap_root / f"test_dino_vs_resnet_probe_seed{seed}.json").read_text(
                    encoding="utf-8"
                )
            )
            for seed in (42, 43, 44)
        },
    }
    return {"images": 2295, "groups": groups, "comparisons": comparisons, "bootstrap": bootstrap}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize confirmatory PlantSeg test runs")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--bootstrap-root", type=Path, default=Path("artifacts/bootstrap"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.runs_root, args.bootstrap_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
