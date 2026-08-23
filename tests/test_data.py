import numpy as np
from PIL import Image

from phytopathology.data import resize_pair


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
