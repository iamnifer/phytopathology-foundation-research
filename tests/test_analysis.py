import pandas as pd

from phytopathology.analyze_predictions import summarize


def test_prediction_summary_groups_by_lesion_size() -> None:
    frame = pd.DataFrame(
        {
            "plant": ["A"] * 8,
            "disease": ["d"] * 8,
            "foreground_iou": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "gt_foreground_fraction": [0.01, 0.02, 0.04, 0.08, 0.16, 0.24, 0.32, 0.4],
            "pred_foreground_fraction": [0.02, 0.02, 0.03, 0.08, 0.12, 0.24, 0.4, 0.4],
        }
    )

    result = summarize(frame)

    assert result["images"] == 8
    assert len(result["by_lesion_size_quartile"]) == 4
    assert result["by_lesion_size_quartile"][0]["mean"] == 0.05
    assert len(result["by_disease_at_least_5_images"]) == 1
