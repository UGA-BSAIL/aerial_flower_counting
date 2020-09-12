"""
Creates the pipeline for data cleaning.
"""


from typing import Any

from kedro.pipeline import Pipeline, node

from .nodes import (
    annotations_as_dataframe,
    drop_out_of_bounds,
    get_frame_sizes,
    remove_unannotated,
)


def create_pipeline(**kwargs: Any) -> Pipeline:
    """
    Create the pipeline for data cleaning.

    Args:
        **kwargs: Ignored for now.

    Returns:
        The pipeline that it created.

    """
    return Pipeline(
        [
            node(remove_unannotated, "cotton_images", "cleaned_annotations"),
            node(
                annotations_as_dataframe,
                "cleaned_annotations",
                "annotations_df",
            ),
            node(
                get_frame_sizes,
                ["annotations_df", "cotton_images"],
                "frame_sizes",
            ),
            node(
                drop_out_of_bounds,
                ["annotations_df", "frame_sizes"],
                "local_annotations",
            ),
        ]
    )
