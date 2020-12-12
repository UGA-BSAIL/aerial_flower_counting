"""
Pipeline for generating automatic annotations.

It extracts patches from unannotated images on CVAT, uses an existing model
to annotate them, and then re-uploads them to CVAT as a new task.
"""

from typing import Any

from kedro.pipeline import Pipeline, node

from .nodes import (
    coerce_patch_shapes,
    predict_patches,
    save_patches_to_disk,
    unannotated_patch_dataset,
    upload_patches,
)


def create_pipeline(**kwargs: Any) -> Pipeline:
    """
    Creates the auto-annotation pipeline.

    Args:
        **kwargs: Ignored.

    Returns:
        The `Pipeline` that it created.

    """
    return Pipeline(
        [
            # Extract patches from unannotated images.
            node(
                unannotated_patch_dataset,
                dict(
                    cvat_task="cotton_part_b_2020_09_19",
                    image_shape="params:cvat_image_shape_phantom_video",
                    batch_size="params:batch_size",
                    num_prefetch_batches="params:num_prefetch_batches",
                    patch_scale="params:annotation_patch_scale",
                    patch_stride="params:annotation_patch_stride",
                    start_frame_num="params:num_initial_frames_to_skip",
                    num_frames="params:num_frames_to_annotate",
                    annotation_interval="params:annotation_interval",
                ),
                "unannotated_patch_data",
            ),
            # Coerce the size of the patches, so all our inputs have the same
            # size.
            node(
                coerce_patch_shapes,
                dict(
                    patches="unannotated_patch_data",
                    desired_shape="params:coerced_patch_shape",
                ),
                "unannotated_resized_patch_data",
            ),
            # Make sure these patches get saved to the disk.
            node(
                save_patches_to_disk,
                dict(
                    patches="unannotated_resized_patch_data",
                    batch_size="params:batch_size",
                ),
                ["unannotated_saved_patch_data", "saved_image_dir"],
            ),
            # Generate the patch annotations.
            node(
                predict_patches,
                dict(
                    model="trained_model",
                    patch_dataset="unannotated_saved_patch_data",
                ),
                "auto_patch_annotations",
            ),
            # Upload to CVAT.
            node(
                upload_patches,
                dict(
                    annotations="auto_patch_annotations",
                    cvat_task="cotton_part_b_2020_09_19",
                ),
                None,
            ),
        ]
    )
