"""
Construction of the master pipeline.
"""

from typing import Any, Dict

from kedro.pipeline import Pipeline

from .pipelines import data_cleaning, data_engineering, eda


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

    return {
        "data_cleaning": cleaning_pipeline,
        "eda": eda_pipeline,
        "data_engineering": engineering_pipeline,
        "__default__": cleaning_pipeline + eda_pipeline + engineering_pipeline,
    }
