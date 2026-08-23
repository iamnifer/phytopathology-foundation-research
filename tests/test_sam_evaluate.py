import numpy as np

from phytopathology.sam_evaluate import bounding_box, positive_point


def test_oracle_prompts_are_inside_and_cover_foreground() -> None:
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:6, 3:9] = True

    point = positive_point(mask)
    box = bounding_box(mask)

    assert mask[int(point[1]), int(point[0])]
    assert box == [3.0, 2.0, 8.0, 5.0]
