import pytest
import torch

from phytopathology.metrics import SegmentationMetrics


def test_dataset_level_miou_ignores_absent_classes() -> None:
    metrics = SegmentationMetrics(num_classes=3)
    metrics.update(torch.tensor([[0, 1], [1, 1]]), torch.tensor([[0, 1], [0, 1]]))
    result = metrics.compute()

    # class 0: 1/2, class 1: 2/3; class 2 is absent and excluded
    assert result["miou"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert result["present_classes"] == 2
    assert result["per_class_iou"][2] is None


def test_ignore_index_does_not_enter_confusion_matrix() -> None:
    metrics = SegmentationMetrics(num_classes=2, ignore_index=255)
    metrics.update(torch.tensor([0, 1, 1]), torch.tensor([0, 1, 255]))
    result = metrics.compute()

    assert result["miou"] == pytest.approx(1.0)
    assert result["pixel_accuracy"] == pytest.approx(1.0)


def test_background_can_be_excluded() -> None:
    metrics = SegmentationMetrics(num_classes=2)
    metrics.update(torch.tensor([0, 0, 0]), torch.tensor([0, 0, 1]))

    assert metrics.compute(exclude_background=True)["miou"] == pytest.approx(0.0)
