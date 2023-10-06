"""
Construction of the master pipeline.
"""

from typing import Any, Dict

from kedro.pipeline import Pipeline

from .pipelines import count_plots, count_plots_v2,  prepare_data


def register_pipelines(**kwargs: Any) -> Dict[str, Pipeline]:
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        A mapping from a pipeline name to a `Pipeline` object.

    """
    plot_counting_pipeline = count_plots.create_pipeline()
    plot_counting_pipeline_v2 = count_plots_v2.create_pipeline()
    prepare_data_pipeline = prepare_data.create_pipeline()

    return {
        "count_plots": plot_counting_pipeline,
        "count_plots_v2": plot_counting_pipeline_v2,
        "prepare_data": prepare_data_pipeline,
    }
