from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate PlantSeg files and mask labels")
    parser.add_argument("root", type=Path)
    parser.add_argument("--metadata-file", default="Metadatav2.csv")
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.root / args.metadata_file)
    required = {"Name", "Label file", "Split"}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise SystemExit(f"Missing columns: {sorted(missing_columns)}")

    split_directories = {"Training": "train", "Validation": "val", "Test": "test"}
    labels: Counter[int] = Counter()
    label_metadata: defaultdict[int, set[str]] = defaultdict(set)
    missing_files: list[str] = []
    checked = 0
    for split_name, rows in frame.groupby("Split"):
        directory = split_directories.get(split_name, split_name.lower())
        if split_name == "Validation" and not (args.root / "images" / directory).is_dir():
            directory = "validation"
        if args.max_samples is not None:
            rows = rows.iloc[: args.max_samples]
        columns = ["Name", "Label file", "Plant", "Disease"]
        iterator = rows[columns].itertuples(index=False, name=None)
        for image_name, mask_name, plant, disease in tqdm(
            iterator, total=len(rows), desc=str(split_name)
        ):
            image_path = args.root / "images" / directory / str(image_name)
            mask_path = args.root / "annotations" / directory / str(mask_name)
            for path in (image_path, mask_path):
                if not path.is_file():
                    missing_files.append(str(path))
            if mask_path.is_file():
                with Image.open(mask_path) as mask:
                    values, counts = np.unique(np.asarray(mask), return_counts=True)
                labels.update(dict(zip(values.tolist(), counts.tolist(), strict=True)))
                for value in values:
                    if value not in (0, 255):
                        label_metadata[int(value)].add(f"{plant}: {disease}")
            checked += 1

    report = {
        "rows_by_split": frame["Split"].value_counts().to_dict(),
        "checked_rows": checked,
        "missing_file_count": len(missing_files),
        "missing_file_examples": missing_files[:20],
        "mask_labels": sorted(labels),
        "mask_label_pixel_counts": dict(sorted(labels.items())),
        "pixel_label_metadata": {
            label: sorted(names) for label, names in sorted(label_metadata.items())
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if missing_files:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
