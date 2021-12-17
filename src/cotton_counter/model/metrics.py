"""
Custom keras metrics.
"""


from typing import Dict, List, Union

from tensorflow import keras


def make_metrics(
    include_count: bool = False,
) -> Dict[str, Union[str, List[keras.metrics.Metric]]]:
    """
    Creates the metrics dictionary to use when compiling a model.

    Args:
        include_count: Whether to include metrics for counting accuracy.
            This only applies to situations where we actually have count
            ground-truth.

    Returns:
        The metrics dictionary that it created.

    """
    metrics = dict(
        # Add standard accuracy metrics for classification.
        has_flower=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.TruePositives(name="tp"),
            keras.metrics.TrueNegatives(name="tn"),
            keras.metrics.FalsePositives(name="fp"),
            keras.metrics.FalseNegatives(name="fn"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ]
    )

    if include_count:
        # Add accuracy metric for counting.
        metrics["count"] = [
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ]

    return metrics
