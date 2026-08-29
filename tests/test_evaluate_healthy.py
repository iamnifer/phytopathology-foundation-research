from __future__ import annotations

import numpy as np

from phytopathology.evaluate_healthy import summarize


def test_healthy_summary_thresholds() -> None:
    result = summarize(np.array([0.0, 0.002, 0.02, 0.2]))
    assert result["images"] == 4
    assert result["images_above_0.001_lesion_fraction"] == 0.75
    assert result["images_above_0.05_lesion_fraction"] == 0.25
