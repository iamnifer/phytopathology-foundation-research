from pathlib import Path

import numpy as np
import torch
from PIL import Image

from phytopathology.evaluate_native import binary_counts, load_binary_target


def test_load_binary_target_preserves_ignore(tmp_path: Path) -> None:
    path = tmp_path / "mask.png"
    Image.fromarray(np.array([[0, 7], [255, 1]], dtype=np.uint8)).save(path)

    target = load_binary_target(path, ignore_index=255)

    assert target.tolist() == [[0, 1], [255, 1]]


def test_binary_counts_ignore_masked_pixels() -> None:
    prediction = torch.tensor([[0, 1], [1, 0]])
    target = torch.tensor([[0, 1], [255, 1]])

    result = binary_counts(prediction, target, ignore_index=255)

    assert result == {
        "true_positive": 1,
        "false_positive": 0,
        "false_negative": 1,
        "valid_pixels": 3,
    }
