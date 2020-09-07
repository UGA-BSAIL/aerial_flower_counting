"""
Creates the pipeline for data cleaning.
"""


from typing import Any

from kedro.pipeline import Pipeline, node

from .nodes import remove_unannotated


def create_pipeline(**kwargs: Any) -> Pipeline:
    """
    Create the pipeline for data cleaning.

    Args:
        **kwargs: Ignored for now.

    Returns:
        The pipeline that it created.

    """
    return Pipeline(
        [node(remove_unannotated, ["cotton_images"], "cleaned_annotations")]
    )
