"""
Pipeline for generating TFRecords from point-annotated data.
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
            # Split into testing and validation.
            node(
                make_splits,
                dict(
                    local_annotations="local_annotations_test",
                    split_fractions="params:test_split_fractions",
                ),
                ["test_annotations", "validate_annotations"],
            ),
            # Shuffle the data.
            node(shuffle, "local_annotations_train", "shuffled_train"),
            node(shuffle, "test_annotations", "shuffled_test"),
            node(shuffle, "validate_annotations", "shuffled_validate"),
            # Convert to TFRecords.
            node(
                generate_tf_records,
                ["shuffled_train", "cotton_part_a"],
                "tfrecord_train",
            ),
            node(
                generate_tf_records,
                ["shuffled_test", "cotton_part_a_test"],
                "tfrecord_test",
            ),
            node(
                generate_tf_records,
                ["shuffled_validate", "cotton_part_a_test"],
                "tfrecord_validate",
            ),
        ]
    )
