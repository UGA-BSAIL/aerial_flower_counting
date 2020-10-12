"""
Defines a pipeline that trains the model.
"""


from functools import partial
from typing import Any

from kedro.pipeline import Pipeline, node

from .nodes import (
    create_model,
    evaluate_model,
    make_callbacks,
    pre_process_dataset,
    train_model,
)

PRE_PROCESS_TAG = "pre_process"
"""
Marks nodes that are part of the pre-processing steps.
"""
INITIAL_TRAIN_TAG = "initial_train"
"""
Marks nodes that are part of the initial training phase.
"""
FINE_TUNING_TAG = "fine_tuning"
"""
Marks nodes that are part of the fine-tuning phase.
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
    # Common parameters shared by pre-processing nodes.
    pre_process_params = dict(
        patch_scale="params:patch_scale",
        map_height="params:density_map_height",
        map_width="params:density_map_width",
        sigma="params:density_map_sigma",
        batch_size="params:batch_size",
        num_prefetch_batches="params:num_prefetch_batches",
        bucket_min_values="params:bucket_min_values",
    )
    # Common parameters shared by training nodes.
    training_params = dict(
        training_data="training_data",
        testing_data="testing_data",
        classify_counts="params:classify_counts",
        callbacks="callbacks",
        validation_frequency="params:validation_frequency",
    )
    # Common parameters shared by evaluation nodes.
    eval_params = dict(
        model="trained_model", classify_counts="params:classify_counts"
    )

    # Training datasets should use random patches, but testing and validation
    # datasets shouldn't.
    pre_process_node_training = partial(
        pre_process_dataset, allow_randomized=True
    )
    pre_process_node_not_training = partial(
        pre_process_dataset, allow_randomized=False
    )

    return Pipeline(
        [
            # Pre-process the data.
            node(
                pre_process_node_training,
                dict(raw_dataset="tfrecord_train", **pre_process_params),
                "training_data",
                tags={PRE_PROCESS_TAG},
            ),
            node(
                pre_process_node_not_training,
                dict(raw_dataset="tfrecord_test", **pre_process_params),
                "testing_data",
                tags={PRE_PROCESS_TAG},
            ),
            node(
                pre_process_node_not_training,
                dict(raw_dataset="tfrecord_validate", **pre_process_params),
                "validation_data",
                tags={PRE_PROCESS_TAG},
            ),
            # Build and train the model.
            node(
                create_model,
                dict(
                    input_width="params:input_width",
                    input_height="params:input_height",
                    patch_scale="params:patch_scale",
                    classify_counts="params:classify_counts",
                    bucket_min_values="params:bucket_min_values",
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
                    max_density_threshold="params:max_density_threshold",
                    bucket_min_values="params:bucket_min_values",
                    classify_counts="params:classify_counts",
                ),
                "callbacks",
                tags={PRE_PROCESS_TAG},
            ),
            # Train initially with just the density map loss.
            node(
                train_model,
                dict(
                    model="initial_model",
                    learning_phases="params:base_learning_phases",
                    **training_params
                ),
                "base_model",
                tags={INITIAL_TRAIN_TAG},
            ),
            # Train again with the count loss as well.
            node(
                train_model,
                dict(
                    model="base_model",
                    learning_phases="params:refined_learning_phases",
                    **training_params
                ),
                "trained_model",
                tags={FINE_TUNING_TAG},
            ),
            # Evaluate model on all datasets.
            node(
                evaluate_model,
                dict(eval_data="training_data", **eval_params),
                "model_report_train",
                tags={EVAL_TAG},
            ),
            node(
                evaluate_model,
                dict(eval_data="testing_data", **eval_params),
                "model_report_test",
                tags={EVAL_TAG},
            ),
            node(
                evaluate_model,
                dict(eval_data="validation_data", **eval_params),
                "model_report_validate",
                tags={EVAL_TAG},
            ),
        ]
    )
