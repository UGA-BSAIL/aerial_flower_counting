"""
Custom keras metrics.
"""


from typing import Dict, Union

from tensorflow import keras


def make_metrics(
    classify_counts: bool = False,
) -> Dict[str, Union[str, keras.metrics.Metric]]:
    """
    Creates the metrics dictionary to use when compiling a model.

    Args:
        classify_counts: Whether we are using the classification count output.

    Returns:
        The metrics dictionary that it created.

    """
    metrics = {"count": "mean_absolute_error"}

    if classify_counts:
        # Add a standard accuracy metric for the classification.
        metrics["discrete_count"] = "sparse_categorical_accuracy"

    return metrics
