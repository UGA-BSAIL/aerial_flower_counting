"""
Defines a pipeline that trains the model.
"""


from typing import Any

from kedro.pipeline import Pipeline, node

from .nodes import (
    create_model,
    evaluate_model,
    pre_process_dataset,
    train_model,
)


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
    )

    return Pipeline(
        [
            # Pre-process the data.
            node(
                pre_process_dataset,
                dict(raw_dataset="tfrecord_train", **pre_process_params,),
                "training_data",
            ),
            node(
                pre_process_dataset,
                dict(raw_dataset="tfrecord_test", **pre_process_params),
                "testing_data",
            ),
            node(
                pre_process_dataset,
                dict(raw_dataset="tfrecord_validate", **pre_process_params),
                "validation_data",
            ),
            # Build and train the model.
            node(
                create_model,
                dict(
                    input_width="params:input_width",
                    input_height="params:input_height",
                    patch_scale="params:patch_scale",
                ),
                "initial_model",
            ),
            node(
                train_model,
                dict(
                    model="initial_model",
                    training_data="training_data",
                    testing_data="testing_data",
                    learning_phases="params:learning_phases",
                ),
                "trained_model",
            ),
            # Evaluate model on all datasets.
            node(
                evaluate_model,
                dict(model="trained_model", eval_data="training_data",),
                "model_report_train",
            ),
            node(
                evaluate_model,
                dict(model="trained_model", eval_data="testing_data",),
                "model_report_test",
            ),
            node(
                evaluate_model,
                dict(model="trained_model", eval_data="validation_data",),
                "model_report_validate",
            ),
        ]
    )
