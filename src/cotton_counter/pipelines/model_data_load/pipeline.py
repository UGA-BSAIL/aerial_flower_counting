"""
A pipeline that sequesters all the nodes for loading data in one place.

This is meant to be used in conjunction with other pipelines.
"""


from functools import partial

from kedro.pipeline import Pipeline, node

from ...model.load_from_dataset import extract_model_input


def create_pipeline(**kwargs):
    # Common parameters shared by pre-processing nodes.
    pre_process_params_no_patch = dict(
        image_shape="params:input_image_shape",
        map_shape="params:density_map_shape",
        sigma="params:density_map_sigma",
        batch_size="params:batch_size",
        num_prefetch_batches="params:num_prefetch_batches",
        bucket_min_values="params:bucket_min_values",
    )
    pre_process_params = dict(
        patch_scale="params:patch_scale", **pre_process_params_no_patch
    )

    # Training datasets should use random patches, but testing and validation
    # datasets shouldn't.
    pre_process_node_training = partial(
        extract_model_input, random_patches=True, shuffle=True
    )
    pre_process_node_not_training = partial(
        extract_model_input, random_patches=False, shuffle=False
    )

    return Pipeline(
        [
            # Pre-process the data.
            node(
                pre_process_node_training,
                dict(raw_dataset="tfrecord_train", **pre_process_params),
                "training_data",
            ),
            node(
                pre_process_node_not_training,
                dict(raw_dataset="tfrecord_test", **pre_process_params),
                "testing_data",
            ),
            node(
                pre_process_node_not_training,
                dict(raw_dataset="tfrecord_validate", **pre_process_params),
                "validation_data",
            ),
            # Create a version of the validation dataset with no patches.
            node(
                pre_process_node_not_training,
                dict(
                    raw_dataset="tfrecord_validate",
                    **pre_process_params_no_patch,
                ),
                "validation_data_no_patches",
            ),
        ]
    )
