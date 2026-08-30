import pandas as pd

from phytopathology.render_dino_sam_panel import mask_iou, select_cases


def test_select_cases_has_distinct_named_examples() -> None:
    frame = pd.DataFrame(
        {
            "image": [f"{index}.jpg" for index in range(8)],
            "foreground_iou": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9],
            "gt_foreground_fraction": [0.01, 0.02, 0.03, 0.04, 0.2, 0.3, 0.4, 0.5],
        }
    )

    cases = select_cases(frame)

    assert [name for name, _ in cases] == [
        "Худший",
        "Медианный",
        "Лучший",
        "Малое поражение",
    ]
    assert len({record["image"] for _, record in cases}) == 4


def test_mask_iou() -> None:
    assert mask_iou(
        prediction=pd.Series([True, False, True]).to_numpy(),
        target=pd.Series([True, True, False]).to_numpy(),
    ) == 1 / 3
