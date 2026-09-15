from pathlib import Path

import pytest

from phytopathology.summarize_confirmatory_test import find_run


def write_run(root: Path, directory: str, seed: int) -> Path:
    path = root / directory
    path.mkdir()
    (path / "config.yaml").write_text(
        f"experiment:\n  name: example\n  seed: {seed}\n",
        encoding="utf-8",
    )
    return path


def test_find_run_uses_experiment_name_and_seed(tmp_path: Path) -> None:
    write_run(tmp_path, "20260101T000000Z_example", 42)
    expected = write_run(tmp_path, "20260102T000000Z_example", 43)
    write_run(tmp_path, "20260103T000000Z_other", 43)

    assert find_run(tmp_path, "example", 43) == expected


def test_find_run_chooses_latest_matching_directory(tmp_path: Path) -> None:
    write_run(tmp_path, "20260101T000000Z_example", 42)
    expected = write_run(tmp_path, "20260102T000000Z_example", 42)

    assert find_run(tmp_path, "example", 42) == expected


def test_find_run_reports_missing_seed(tmp_path: Path) -> None:
    write_run(tmp_path, "20260101T000000Z_example", 42)

    with pytest.raises(FileNotFoundError, match="seed=43"):
        find_run(tmp_path, "example", 43)
