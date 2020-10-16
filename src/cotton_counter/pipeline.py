"""
Construction of the master pipeline.
"""

from typing import Any, Dict

from kedro.pipeline import Pipeline

from .pipelines import (
    data_cleaning,
    data_engineering,
    eda,
    model_data_load,
    model_evaluation,
    model_training,
)


def create_pipelines(**kwargs: Any) -> Dict[str, Pipeline]:
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        A mapping from a pipeline name to a `Pipeline` object.

    """
    cleaning_pipeline = data_cleaning.create_pipeline()
    eda_pipeline = eda.create_pipeline()
    engineering_pipeline = data_engineering.create_pipeline()
    training_pipeline = model_training.create_pipeline()
    evaluation_pipeline = model_evaluation.create_pipeline()
    data_loading_pipeline = model_data_load.create_pipeline()

    return {
        "model_training": data_loading_pipeline + training_pipeline,
        "model_evaluation": data_loading_pipeline + evaluation_pipeline,
        "prepare_data": cleaning_pipeline
        + eda_pipeline
        + engineering_pipeline,
        "__default__": data_loading_pipeline
        + training_pipeline
        + evaluation_pipeline,
    }
