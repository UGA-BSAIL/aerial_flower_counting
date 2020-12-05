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
    remove_with_attributes,
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
            # Clean the training data.
            node(
                remove_unannotated,
                "cotton_part_a",
                "non_empty_annotations_train",
            ),
            node(
                remove_with_attributes,
                dict(
                    annotations="non_empty_annotations_train",
                    cvat_task="cotton_part_a",
                    attributes="params:remove_with_attributes",
                ),
                "cleaned_annotations_train",
            ),
            node(
                annotations_as_dataframe,
                "cleaned_annotations_train",
                "annotations_df_train",
            ),
            node(
                get_frame_sizes,
                ["annotations_df_train", "cotton_part_a"],
                "frame_sizes_train",
            ),
            node(
                drop_out_of_bounds,
                ["annotations_df_train", "frame_sizes_train"],
                "local_annotations_train",
            ),
            # Clean the testing data.
            node(
                remove_unannotated,
                "cotton_part_a_test",
                "non_empty_annotations_test",
            ),
            node(
                remove_with_attributes,
                dict(
                    annotations="non_empty_annotations_test",
                    cvat_task="cotton_part_a_test",
                    attributes="params:remove_with_attributes",
                ),
                "cleaned_annotations_test",
            ),
            node(
                annotations_as_dataframe,
                "cleaned_annotations_test",
                "annotations_df_test",
            ),
            node(
                get_frame_sizes,
                ["annotations_df_test", "cotton_part_a_test"],
                "frame_sizes_test",
            ),
            node(
                drop_out_of_bounds,
                ["annotations_df_test", "frame_sizes_test"],
                "local_annotations_test",
            ),
        ]
    )
