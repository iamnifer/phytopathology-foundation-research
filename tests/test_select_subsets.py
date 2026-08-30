import numpy as np

from phytopathology.select_subsets import kmeans_representatives


def test_kmeans_representatives_choose_one_unique_member_per_cluster() -> None:
    descriptors = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
            [-1.0, 0.0],
            [-0.99, -0.01],
        ],
        dtype=np.float32,
    )

    selected = kmeans_representatives(
        descriptors, count=3, seed=42, iterations=5, restarts=1
    )

    assert len(selected) == 3
    assert len(set(selected.tolist())) == 3
    assert set(selected).issubset(range(len(descriptors)))
