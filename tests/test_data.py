import numpy as np
import pandas as pd
import pytest
from PIL import Image

from phytopathology.data import resize_pair
from phytopathology.select_subsets import random_order, stratified_order
from phytopathology.train import small_lesion_sampling_weights


def test_pad_resize_preserves_geometry_and_mask_labels() -> None:
    image = Image.new("RGB", (8, 4), color=(10, 20, 30))
    mask_array = np.zeros((4, 8), dtype=np.uint8)
    mask_array[:, 2:6] = 3
    mask = Image.fromarray(mask_array)

    resized_image, resized_mask = resize_pair(image, mask, 16, "pad")
    result = np.asarray(resized_mask)

    assert resized_image.size == (16, 16)
    assert resized_mask.size == (16, 16)
    assert set(np.unique(result)) == {0, 3}
    assert np.all(result[:4] == 0)
    assert np.all(result[-4:] == 0)


def test_small_lesion_sampling_upweights_only_lowest_positive_quartile() -> None:
    fractions = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.20])

    weights, threshold = small_lesion_sampling_weights(fractions, 0.25, 2.0)

    assert threshold == pytest.approx(0.02)
    assert weights.tolist() == [1.0, 2.0, 2.0, 1.0, 1.0, 1.0]


def test_small_lesion_sampling_rejects_invalid_parameters() -> None:
    fractions = np.array([0.01, 0.10])

    with pytest.raises(ValueError, match="quantile"):
        small_lesion_sampling_weights(fractions, 1.0, 2.0)
    with pytest.raises(ValueError, match="factor"):
        small_lesion_sampling_weights(fractions, 0.25, 0.5)


def test_subset_orders_are_reproducible_and_stratified() -> None:
    names = np.array(["a", "b", "c", "d", "e", "f"])
    metadata = pd.DataFrame(
        {"Name": names, "Disease": ["x", "x", "x", "y", "y", "y"]}
    )

    assert np.array_equal(random_order(6, 42), random_order(6, 42))
    order = stratified_order(names, metadata, 42)

    assert set(order[:2]) & {0, 1, 2}
    assert set(order[:2]) & {3, 4, 5}
