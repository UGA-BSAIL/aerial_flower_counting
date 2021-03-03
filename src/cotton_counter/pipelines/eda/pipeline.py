"""
Creates the pipeline for EDA.
"""


from typing import Any

from kedro.pipeline import Pipeline, node

from .nodes import (
    annotation_histogram,
    annotation_spatial_dist,
    visualize_ground_truth,
)


def create_pipeline(**kwargs: Any):
    """
    Creates the pipeline for EDA.
    Args:
        **kwargs: Ignored for now.

    Returns:
        The pipeline that it create.

    """
    return Pipeline(
        [
            # Perform EDA for training data.
            node(
                annotation_histogram,
                "local_annotations_train",
                "eda_annotation_histogram_train",
            ),
            node(
                annotation_spatial_dist,
                "local_annotations_train",
                "eda_annotation_spatial_dist_train",
            ),
            # Perform EDA for testing and validation data.
            node(
                annotation_histogram,
                "local_annotations_test",
                "eda_annotation_histogram_test",
            ),
            node(
                annotation_spatial_dist,
                "local_annotations_test",
                "eda_annotation_spatial_dist_test",
            ),
            node(
                visualize_ground_truth,
                "validation_data_raw_annotations",
                "eda_validation_ground_truth",
            ),
        ]
    )
