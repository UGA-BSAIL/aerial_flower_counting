"""
Creates the pipeline for EDA.
"""


from typing import Any

from kedro.pipeline import Pipeline, node

from .nodes import annotation_histogram, annotation_spatial_dist


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
            node(
                annotation_histogram,
                "local_annotations",
                "eda_annotation_histogram",
            ),
            node(
                annotation_spatial_dist,
                "local_annotations",
                "eda_annotation_spatial_dist",
            ),
        ]
    )
