"""
Defines a pipeline that trains the model.
"""


from typing import Any

from kedro.pipeline import Pipeline, node

from .nodes import train_model


def create_pipeline(**kwargs: Any):
    """
    Defines a pipeline that trains the model.

    Args:
        kwargs: Ignored by this function.
    """
    return Pipeline(
        [
            node(
                train_model,
                dict(
                    training_data="tfrecord_train",
                    testing_data="tfrecord_test",
                    input_width="params:input_width",
                    input_height="params:input_height",
                    patch_scale="params:patch_scale",
                    map_height="params:density_map_height",
                    map_width="params:density_map_width",
                    sigma="params:density_map_sigma",
                    batch_size="params:batch_size",
                    num_prefetch_batches="params:num_prefetch_batches",
                    learning_rate="params:learning_rate",
                    momentum="params:momentum",
                ),
                None,
            )
        ]
    )
