"""
Defines a pipeline that trains the model.
"""


from typing import Any

from kedro.pipeline import Pipeline, node

from .nodes import (
    create_model,
    make_callbacks,
    set_check_numerics,
    train_model,
)

PRE_PROCESS_TAG = "pre_process"
"""
Marks nodes that are part of the pre-processing steps.
"""
TRAIN_TAG = "train_tag"
"""
Marks nodes that are part of the training phase.
"""
EVAL_TAG = "evaluation"
"""
Marks nodes that are part of the evaluation phase.
"""


def create_pipeline(**kwargs: Any):
    """
    Defines a pipeline that trains the model.

    Args:
        kwargs: Ignored by this function.
    """
    # Common parameters shared by training nodes.
    training_params = dict(
        training_data_manager="dataset_manager",
        testing_data="testing_data",
        callbacks="callbacks",
        validation_frequency="params:validation_frequency",
        rebalance_frequency="params:rebalance_frequency",
        tensorboard_output_dir="params:tensorboard_output_dir",
        batch_size="params:batch_size",
        tag_fraction="params:tag_fraction",
        focal_loss_alpha="params:focal_loss_alpha",
        focal_loss_gamma="params:focal_loss_gamma",
    )

    return Pipeline(
        [
            # Build and train the model.
            node(set_check_numerics, "params:enable_numeric_checks", None),
            node(
                create_model,
                dict(
                    input_image_shape="params:input_image_shape",
                    patch_scale="params:patch_scale",
                    num_loss_scales="params:num_loss_scales",
                    initial_output_bias="initial_output_bias",
                    focal_loss_alpha="params:focal_loss_alpha",
                    focal_loss_gamma="params:focal_loss_gamma",
                ),
                "initial_model",
                tags={PRE_PROCESS_TAG},
            ),
            # Create the callbacks to use.
            node(
                make_callbacks,
                dict(
                    model="initial_model",
                    testing_data="testing_data",
                    tensorboard_output_dir="params:tensorboard_output_dir",
                    histogram_frequency="params:histogram_frequency",
                    visualization_period="params:visualization_period",
                    num_per_batch_to_visualize="params"
                    ":num_per_batch_to_visualize",
                    max_num_batches_to_visualize="params"
                    ":max_num_batches_to_visualize",
                ),
                "callbacks",
                tags={PRE_PROCESS_TAG},
            ),
            # Train the model.
            node(
                train_model,
                dict(
                    model="initial_model",
                    learning_phases="params:learning_phases",
                    **training_params
                ),
                "trained_model",
                tags={TRAIN_TAG},
            ),
        ]
    )
