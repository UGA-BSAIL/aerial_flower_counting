"""
A pipeline that re-uploads existing annotations to CVAT.
"""

from kedro.pipeline import Pipeline, node

from ..auto_annotation.nodes import save_patches_to_disk, upload_patches
from .nodes import annotations_to_dataframe, combine_inputs_and_targets


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                combine_inputs_and_targets,
                "tagged_patch_data_positive",
                "combined_example_patch_data",
            ),
            node(
                save_patches_to_disk,
                dict(
                    patches="combined_example_patch_data",
                    batch_size="params:batch_size",
                ),
                ["patch_data_with_saving", "temporary_image_dir"],
            ),
            node(
                annotations_to_dataframe,
                "patch_data_with_saving",
                "gt_annotation_data",
            ),
            node(
                upload_patches,
                dict(
                    annotations="gt_annotation_data",
                    # It doesn't actually matter what task we use, because it
                    # will only copy the credentials.
                    cvat_task="cotton_part_a",
                ),
                None,
            ),
        ]
    )
