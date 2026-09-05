from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the healthy PlantWild v1 manifest and optionally download images"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repository", default="Voxel51/PlantWild")
    parser.add_argument("--download-split", choices=("train", "val", "test"))
    return parser.parse_args()


def healthy_manifest(payload: dict[str, object]) -> list[dict[str, str]]:
    records = []
    for sample in payload["samples"]:
        label = str(sample["ground_truth"]["label"])
        if sample.get("dataset_version") != "v1" or not label.lower().endswith(" leaf"):
            continue
        records.append(
            {
                "path": str(sample["filepath"]),
                "label": label,
                "split": str(sample["split"]),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    samples_path = args.dataset_root / "samples.json"
    if not samples_path.is_file():
        from huggingface_hub import snapshot_download

        snapshot_download(
            args.repository,
            repo_type="dataset",
            allow_patterns=["samples.json", "metadata.json", "fiftyone.yml"],
            local_dir=args.dataset_root,
        )
    records = healthy_manifest(json.loads(samples_path.read_text(encoding="utf-8")))
    manifest_path = args.dataset_root / "healthy_manifest.json"
    manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    if args.download_split:
        from huggingface_hub import snapshot_download

        paths = [record["path"] for record in records if record["split"] == args.download_split]
        snapshot_download(
            args.repository,
            repo_type="dataset",
            allow_patterns=paths,
            local_dir=args.dataset_root,
        )
    counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "val", "test")
    }
    print(json.dumps({"images": len(records), "split_counts": counts}))


if __name__ == "__main__":
    main()
