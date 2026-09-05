from phytopathology.prepare_plantwild_healthy import healthy_manifest


def test_healthy_manifest_keeps_v1_leaf_classes() -> None:
    payload = {
        "samples": [
            {
                "filepath": "data/healthy.jpg",
                "dataset_version": "v1",
                "split": "test",
                "ground_truth": {"label": "apple leaf"},
            },
            {
                "filepath": "data/disease.jpg",
                "dataset_version": "v1",
                "split": "test",
                "ground_truth": {"label": "apple black rot"},
            },
            {
                "filepath": "data/v2.jpg",
                "dataset_version": "v2",
                "split": "test",
                "ground_truth": {"label": "banana leaf"},
            },
        ]
    }

    assert healthy_manifest(payload) == [
        {"path": "data/healthy.jpg", "label": "apple leaf", "split": "test"}
    ]
