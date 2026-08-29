from __future__ import annotations

import pandas as pd

from phytopathology.analyze_subsets import summarize_subset


def test_subset_summary_reports_distribution_shift() -> None:
    pool = pd.DataFrame(
        {
            "Disease": ["a", "a", "b", "b"],
            "Plant": ["x", "x", "y", "y"],
            "Mask ratio": [0.001, 0.01, 0.1, 0.2],
            "source": ["s", "s", "t", "t"],
        }
    )
    summary = summarize_subset(pool, pool.iloc[:2])
    assert summary["images"] == 2
    assert summary["unique_diseases"] == 1
    assert summary["disease_distribution_total_variation"] == 0.5
