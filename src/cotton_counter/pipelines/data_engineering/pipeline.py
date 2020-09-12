"""
Pipeline for data engineering tasks.
"""


from typing import Any

from kedro.pipeline import Pipeline, node

from .nodes import generate_tf_records, make_splits, shuffle


def create_pipeline(**kwargs: Any):
    """
    Creates the data engineering pipeline.

    Args:
        **kwargs: Will be ignored.

    Returns:
        The pipeline that it created.

    """
    return Pipeline(
        [
            # Split into training, testing and validation.
            node(
                make_splits,
                dict(
                    local_annotations="local_annotations",
                    train_fraction="params:train_fraction",
                    test_fraction="params:test_fraction",
                ),
                [
                    "train_annotations",
                    "test_annotations",
                    "validate_annotations",
                ],
            ),
            # Shuffle the data.
            node(shuffle, "train_annotations", "shuffled_train"),
            node(shuffle, "test_annotations", "shuffled_test"),
            node(shuffle, "validate_annotations", "shuffled_validate"),
            # Convert to TFRecords.
            node(
                generate_tf_records,
                ["shuffled_train", "cotton_images"],
                "tfrecord_train",
            ),
            node(
                generate_tf_records,
                ["shuffled_test", "cotton_images"],
                "tfrecord_test",
            ),
            node(
                generate_tf_records,
                ["shuffled_validate", "cotton_images"],
                "tfrecord_validate",
            ),
        ]
    )
