from __future__ import annotations

import torch


class SegmentationMetrics:
    """Dataset-level semantic-segmentation metrics based on a confusion matrix."""

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.detach().to("cpu", dtype=torch.int64).flatten()
        target = target.detach().to("cpu", dtype=torch.int64).flatten()
        valid = (
            (target != self.ignore_index)
            & (target >= 0)
            & (target < self.num_classes)
            & (prediction >= 0)
            & (prediction < self.num_classes)
        )
        encoded = target[valid] * self.num_classes + prediction[valid]
        counts = torch.bincount(encoded, minlength=self.num_classes**2)
        self.confusion += counts.reshape(self.num_classes, self.num_classes)

    def compute(self, exclude_background: bool = False) -> dict[str, object]:
        matrix = self.confusion.to(torch.float64)
        intersection = matrix.diag()
        target_count = matrix.sum(dim=1)
        prediction_count = matrix.sum(dim=0)
        union = target_count + prediction_count - intersection
        iou = torch.full_like(union, torch.nan)
        present = union > 0
        iou[present] = intersection[present] / union[present]
        start = 1 if exclude_background else 0
        selected = iou[start:]
        mean_iou = (
            torch.nanmean(selected).item()
            if not torch.isnan(selected).all()
            else float("nan")
        )
        total = matrix.sum()
        pixel_accuracy = (intersection.sum() / total).item() if total else float("nan")
        return {
            "miou": mean_iou,
            "pixel_accuracy": pixel_accuracy,
            "per_class_iou": [None if torch.isnan(value) else value.item() for value in iou],
            "present_classes": int(present[start:].sum().item()),
        }
