from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from phytopathology.summarize_probes import summarize


def write_run(root: Path, name: str, seed: int, iou: float) -> None:
    run = root / f"20260918T00000{seed}Z_{name}"
    run.mkdir()
    (run / "config.yaml").write_text(
        yaml.safe_dump({"experiment": {"seed": seed}}), encoding="utf-8"
    )
    validation = {
        "foreground_iou": iou,
        "miou": iou + 0.1,
        "foreground_dice": iou + 0.2,
        "foreground_precision": iou + 0.3,
        "foreground_recall": iou + 0.4,
        "pixel_average_precision": iou + 0.5,
    }
    records = [
        {"epoch": epoch, "validation": {**validation, "foreground_iou": iou - 0.01}}
        for epoch in range(1, 30)
    ]
    records.append({"epoch": 30, "validation": validation})
    (run / "metrics.jsonl").write_text(
        "".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8"
    )
    (run / "model.json").write_text(
        json.dumps({"total": 100, "trainable": 2}), encoding="utf-8"
    )
    (run / "runtime.json").write_text(
        json.dumps({"total_seconds": 10.0, "peak_cuda_memory_bytes": 20}), encoding="utf-8"
    )


def test_summarize_paired_probes(tmp_path: Path) -> None:
    for seed, offset in ((42, 0.0), (43, 0.01), (44, 0.02)):
        write_run(tmp_path, "resnet50_binary_linear_probe_cosine", seed, 0.4 + offset)
        write_run(tmp_path, "dinov3_vitb16_binary_linear_probe_cosine", seed, 0.6 + offset)

    result = summarize(tmp_path)

    assert result["groups"]["resnet_matched_24"]["aggregate"]["foreground_iou"][
        "mean"
    ] == pytest.approx(0.41)
    paired = result["paired_dino_minus_resnet_24"]
    assert paired["seeds"] == [42, 43, 44]
    assert paired["mean"] == pytest.approx(0.2)
