from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze selected PlantSeg subsets")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=16)
    return parser.parse_args()


def total_variation(pool: pd.Series, selected: pd.Series) -> float:
    labels = pool.index.union(selected.index)
    difference = pool.reindex(labels, fill_value=0) - selected.reindex(labels, fill_value=0)
    return float(0.5 * difference.abs().sum())


def summarize_subset(pool: pd.DataFrame, selected: pd.DataFrame) -> dict[str, object]:
    ratios = selected["Mask ratio"].astype(float)
    pool_disease = pool["Disease"].astype(str).value_counts(normalize=True)
    selected_disease = selected["Disease"].astype(str).value_counts(normalize=True)
    pool_plant = pool["Plant"].astype(str).value_counts(normalize=True)
    selected_plant = selected["Plant"].astype(str).value_counts(normalize=True)
    pool_source = pool["source"].astype(str).value_counts(normalize=True)
    selected_source = selected["source"].astype(str).value_counts(normalize=True)
    return {
        "images": len(selected),
        "unique_diseases": int(selected["Disease"].nunique()),
        "unique_plants": int(selected["Plant"].nunique()),
        "unique_sources": int(selected["source"].nunique()),
        "disease_distribution_total_variation": total_variation(pool_disease, selected_disease),
        "plant_distribution_total_variation": total_variation(pool_plant, selected_plant),
        "source_distribution_total_variation": total_variation(pool_source, selected_source),
        "mask_ratio_mean": float(ratios.mean()),
        "mask_ratio_median": float(ratios.median()),
        "mask_ratio_q10": float(ratios.quantile(0.1)),
        "mask_ratio_q90": float(ratios.quantile(0.9)),
        "mask_ratio_below_0_005": float((ratios < 0.005).mean()),
        "sources": selected["source"].value_counts().head(10).to_dict(),
    }


def render_contact_sheet(
    selected: pd.DataFrame, image_dir: Path, count: int, title: str, output: Path
) -> None:
    count = min(count, len(selected))
    columns = min(8, count)
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(2.2 * columns, 2.4 * rows), squeeze=False)
    for axis, (_, sample) in zip(axes.flat, selected.iloc[:count].iterrows(), strict=False):
        with Image.open(image_dir / str(sample["Name"])) as source:
            axis.imshow(source.convert("RGB"))
        sample_title = f"{sample['Plant']}\nпоражение {100 * float(sample['Mask ratio']):.1f}%"
        axis.set_title(sample_title, fontsize=8)
        axis.axis("off")
    for axis in axes.flat[count:]:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata)
    pool = metadata[metadata["Split"] == "Training"].copy()
    pool["source"] = pool["URL"].fillna("").map(lambda value: urlparse(str(value)).netloc)
    indexed = pool.set_index("Name", drop=False)
    summaries: dict[str, object] = {}
    composition_rows = []
    for manifest_path in args.manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        names = manifest["samples"]
        missing = set(names).difference(indexed.index)
        if missing:
            raise ValueError(f"Unknown samples in {manifest_path}: {sorted(missing)[:5]}")
        selected = indexed.loc[names].copy()
        seed_suffix = f"_seed{manifest['seed']}" if manifest.get("seed") is not None else ""
        key = f"{manifest['method']}{seed_suffix}_{manifest['budget_fraction']:g}"
        summaries[key] = summarize_subset(pool, selected)
        for disease, count in selected["Disease"].value_counts().items():
            composition_rows.append(
                {"subset": key, "disease": disease, "images": int(count)}
            )
        render_contact_sheet(
            selected,
            args.data_root / "images" / "train",
            args.num_images,
            {
                "random": "Случайный отбор",
                "stratified": "Стратифицированный случайный отбор",
                "farthest": "Жадный отбор наиболее удалённых изображений",
                "kmeans": "Представители кластеров сферического k-means",
            }[manifest["method"]],
            args.output_dir / f"{key}_contact_sheet.png",
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(composition_rows).to_csv(args.output_dir / "disease_composition.csv", index=False)
    print(json.dumps(summaries, ensure_ascii=False))


if __name__ == "__main__":
    main()
