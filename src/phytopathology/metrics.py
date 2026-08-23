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
        result = {
            "miou": mean_iou,
            "pixel_accuracy": pixel_accuracy,
            "per_class_iou": [None if torch.isnan(value) else value.item() for value in iou],
            "present_classes": int(present[start:].sum().item()),
        }
        if self.num_classes == 2:
            true_positive = matrix[1, 1]
            false_positive = matrix[0, 1]
            false_negative = matrix[1, 0]
            result.update(
                {
                    "foreground_iou": _safe_ratio(
                        true_positive, true_positive + false_positive + false_negative
                    ),
                    "foreground_dice": _safe_ratio(
                        2 * true_positive, 2 * true_positive + false_positive + false_negative
                    ),
                    "foreground_precision": _safe_ratio(
                        true_positive, true_positive + false_positive
                    ),
                    "foreground_recall": _safe_ratio(
                        true_positive, true_positive + false_negative
                    ),
                }
            )
        return result


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    return (numerator / denominator).item() if denominator else float("nan")


class BinaryAveragePrecision:
    """Streaming approximation of pixel AP using fixed probability bins."""

    def __init__(self, bins: int = 2048, ignore_index: int = 255) -> None:
        self.bins = bins
        self.ignore_index = ignore_index
        self.positive = torch.zeros(bins, dtype=torch.int64)
        self.negative = torch.zeros(bins, dtype=torch.int64)

    def update(self, foreground_probability: torch.Tensor, target: torch.Tensor) -> None:
        scores = foreground_probability.detach().flatten()
        target = target.detach().flatten()
        valid = target != self.ignore_index
        scores, target = scores[valid], target[valid]
        indices = (scores * self.bins).to(torch.int64).clamp_(0, self.bins - 1)
        positive = torch.bincount(indices[target == 1], minlength=self.bins)
        negative = torch.bincount(indices[target == 0], minlength=self.bins)
        self.positive += positive.cpu()
        self.negative += negative.cpu()

    def compute(self) -> float:
        positives = self.positive.flip(0).cumsum(0).to(torch.float64)
        negatives = self.negative.flip(0).cumsum(0).to(torch.float64)
        total_positives = positives[-1]
        if not total_positives:
            return float("nan")
        precision = positives / (positives + negatives).clamp_min(1)
        recall = positives / total_positives
        recall_increment = torch.diff(recall, prepend=torch.zeros(1, dtype=recall.dtype))
        return (precision * recall_increment).sum().item()


class BinaryThresholdSweep:
    """Dataset-level binary metrics over probability thresholds using histograms."""

    def __init__(self, bins: int = 2048, ignore_index: int = 255) -> None:
        if bins < 2:
            raise ValueError("bins must be at least 2")
        self.bins = bins
        self.ignore_index = ignore_index
        self.positive = torch.zeros(bins, dtype=torch.int64)
        self.negative = torch.zeros(bins, dtype=torch.int64)

    def update(self, foreground_probability: torch.Tensor, target: torch.Tensor) -> None:
        scores = foreground_probability.detach().flatten()
        target = target.detach().flatten()
        valid = target != self.ignore_index
        scores, target = scores[valid], target[valid]
        indices = (scores * self.bins).to(torch.int64).clamp_(0, self.bins - 1)
        self.positive += torch.bincount(
            indices[target == 1], minlength=self.bins
        ).cpu()
        self.negative += torch.bincount(
            indices[target == 0], minlength=self.bins
        ).cpu()

    def compute(self, thresholds: list[float]) -> list[dict[str, float]]:
        positive_above = self.positive.flip(0).cumsum(0).flip(0).to(torch.float64)
        negative_above = self.negative.flip(0).cumsum(0).flip(0).to(torch.float64)
        total_positive = self.positive.sum().to(torch.float64)
        total_negative = self.negative.sum().to(torch.float64)
        rows = []
        for threshold in thresholds:
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("thresholds must be in [0, 1]")
            index = min(int(threshold * self.bins), self.bins - 1)
            true_positive = positive_above[index]
            false_positive = negative_above[index]
            false_negative = total_positive - true_positive
            true_negative = total_negative - false_positive
            foreground_iou = _safe_ratio(
                true_positive, true_positive + false_positive + false_negative
            )
            background_iou = _safe_ratio(
                true_negative, true_negative + false_positive + false_negative
            )
            rows.append(
                {
                    "threshold": float(threshold),
                    "miou": (foreground_iou + background_iou) / 2,
                    "foreground_iou": foreground_iou,
                    "background_iou": background_iou,
                    "foreground_dice": _safe_ratio(
                        2 * true_positive,
                        2 * true_positive + false_positive + false_negative,
                    ),
                    "foreground_precision": _safe_ratio(
                        true_positive, true_positive + false_positive
                    ),
                    "foreground_recall": _safe_ratio(
                        true_positive, true_positive + false_negative
                    ),
                }
            )
        return rows
