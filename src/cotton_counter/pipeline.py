"""
Construction of the master pipeline.
"""

from typing import Any, Dict

from kedro.pipeline import Pipeline

from .pipelines import (
    auto_annotation,
    build_tfrecords_patches,
    build_tfrecords_points,
    data_cleaning,
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
    points_pipeline = build_tfrecords_points.create_pipeline()
    patches_pipeline = build_tfrecords_patches.create_pipeline()
    training_pipeline = model_training.create_pipeline()
    evaluation_pipeline = model_evaluation.create_pipeline()
    data_loading_pipeline = model_data_load.create_pipeline()
    auto_annotation_pipeline = auto_annotation.create_pipeline()

    return {
        "model_training": data_loading_pipeline + training_pipeline,
        "model_evaluation": data_loading_pipeline + evaluation_pipeline,
        "auto_annotation": auto_annotation_pipeline,
        "prepare_data": cleaning_pipeline
        + eda_pipeline
        + points_pipeline
        + patches_pipeline,
        "prepare_patches": patches_pipeline,
        "__default__": data_loading_pipeline
        + training_pipeline
        + evaluation_pipeline,
    }
