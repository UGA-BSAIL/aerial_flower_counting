"""
A pipeline that sequesters all the nodes for loading data in one place.

This is meant to be used in conjunction with other pipelines.
"""


from functools import partial

from kedro.pipeline import Pipeline, node

from ...model.dataset_io import (
    inputs_and_targets_from_dataset,
    inputs_and_targets_from_patch_dataset,
)
from .nodes import (
    add_sub_patch_target,
    calculate_output_bias,
    combine_point_and_tag_datasets,
)


def create_pipeline(**kwargs):
    # Parameters for tagged patch datasets.
    pre_process_params_tagged_patches = dict(
        batch_size="params:batch_size",
        num_prefetch_batches="params:num_prefetch_batches",
    )
    # Parameters for point datasets with no patching.
    pre_process_params_no_patch = dict(
        image_shape="params:input_image_shape",
        **pre_process_params_tagged_patches,
    )
    # Parameters for point datasets with patching.
    pre_process_params = dict(
        patch_scale="params:patch_scale", **pre_process_params_no_patch
    )

    # Training datasets should use random patches, but testing and validation
    # datasets shouldn't.
    pre_process_node_training = partial(
        inputs_and_targets_from_dataset, random_patches=True,
    )
    pre_process_node_not_training = partial(
        inputs_and_targets_from_dataset, random_patches=False,
    )
    # Sometimes we want to include raw counts.
    pre_process_node_with_counts = partial(
        pre_process_node_not_training, include_counts=True,
    )

    return Pipeline(
        [
            # Pre-process the data.
            node(
                pre_process_node_training,
                dict(raw_dataset="tfrecord_train", **pre_process_params),
                "training_data_no_sub_patch_target",
            ),
            node(
                pre_process_node_not_training,
                dict(raw_dataset="tfrecord_test", **pre_process_params),
                "testing_data_no_sub_patch_target",
            ),
            node(
                pre_process_node_not_training,
                dict(raw_dataset="tfrecord_validate", **pre_process_params),
                "validation_data_no_sub_patch_target",
            ),
            # Create a version of the validation dataset with no patches.
            node(
                pre_process_node_with_counts,
                dict(
                    raw_dataset="tfrecord_validate",
                    **pre_process_params_no_patch,
                ),
                "validation_data_no_patches",
            ),
            # Pre-process the tagged patch dataset.
            node(
                inputs_and_targets_from_patch_dataset,
                dict(
                    raw_dataset="tfrecord_tagged_patches_positive",
                    **pre_process_params_tagged_patches,
                ),
                "tagged_patch_data_positive",
            ),
            node(
                inputs_and_targets_from_patch_dataset,
                dict(
                    raw_dataset="tfrecord_tagged_patches_negative",
                    **pre_process_params_tagged_patches,
                ),
                "tagged_patch_data_negative",
            ),
            # Pre-process the tag dataset.
            node(
                combine_point_and_tag_datasets,
                dict(
                    point_dataset="training_data",
                    tag_dataset_positive="tagged_patch_data_positive",
                    tag_dataset_negative="tagged_patch_data_negative",
                    tag_fraction="params:tag_fraction",
                    num_positive_patches="params:num_positive_patches",
                    num_negative_patches="params:num_negative_patches",
                    batch_size="params:batch_size",
                ),
                "combined_training_data_no_sub_patch_target",
            ),
            # Add sub-patch targets to everything that needs them.
            node(
                add_sub_patch_target,
                "training_data_no_sub_patch_target",
                "training_data",
            ),
            node(
                add_sub_patch_target,
                "testing_data_no_sub_patch_target",
                "testing_data",
            ),
            node(
                add_sub_patch_target,
                "validation_data_no_sub_patch_target",
                "validation_data",
            ),
            node(
                add_sub_patch_target,
                "combined_training_data_no_sub_patch_target",
                "combined_training_data",
            ),
            # Calculate the initial output bias.
            node(
                calculate_output_bias,
                dict(
                    point_positive_fraction="params:point_positive_fraction",
                    tag_fraction="params:tag_fraction",
                    num_positive_patches="params:num_positive_patches",
                    num_negative_patches="params:num_negative_patches",
                ),
                "initial_output_bias",
            ),
        ]
    )
