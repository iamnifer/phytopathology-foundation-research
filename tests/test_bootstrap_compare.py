from __future__ import annotations

import pandas as pd
import pytest

from phytopathology.bootstrap_compare import paired_bootstrap


def frame(images: list[str], tp: list[int], fp: list[int], fn: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {"image": images, "true_positive": tp, "false_positive": fp, "false_negative": fn}
    )


def test_paired_bootstrap_detects_consistent_improvement() -> None:
    baseline = frame(["a", "b", "c"], [5, 5, 5], [5, 5, 5], [5, 5, 5])
    candidate = frame(["a", "b", "c"], [9, 9, 9], [1, 1, 1], [1, 1, 1])
    result = paired_bootstrap(baseline, candidate, iterations=100, seed=1)
    assert result["difference"] > 0
    assert result["ci_95_lower"] > 0


def test_paired_bootstrap_rejects_different_image_sets() -> None:
    baseline = frame(["a"], [1], [0], [0])
    candidate = frame(["b"], [1], [0], [0])
    with pytest.raises(ValueError, match="image sets differ"):
        paired_bootstrap(baseline, candidate, iterations=10)
