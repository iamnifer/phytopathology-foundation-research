import numpy as np
import torch

from phytopathology.render_error_atlas import error_overlay


def test_error_overlay_colors_tp_fp_and_fn() -> None:
    image = np.zeros((2, 2, 3), dtype=np.float64)
    prediction = torch.tensor([[1, 1], [0, 0]])
    target = torch.tensor([[1, 0], [1, 0]])

    result = error_overlay(image, prediction, target, alpha=1.0)

    assert np.allclose(result[0, 0], (0.10, 0.85, 0.10))
    assert np.allclose(result[0, 1], (1.00, 0.10, 0.10))
    assert np.allclose(result[1, 0], (0.10, 0.35, 1.00))
    assert np.allclose(result[1, 1], (0.00, 0.00, 0.00))


def test_error_overlay_leaves_ignored_pixels_uncolored() -> None:
    image = np.zeros((1, 1, 3), dtype=np.float64)
    prediction = torch.tensor([[1]])
    target = torch.tensor([[255]])

    result = error_overlay(image, prediction, target, alpha=1.0)

    assert np.allclose(result[0, 0], (0.00, 0.00, 0.00))
