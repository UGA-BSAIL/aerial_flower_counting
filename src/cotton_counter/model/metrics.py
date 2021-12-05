"""
Custom keras metrics.
"""


from typing import Dict, List, Union

from tensorflow import keras


def make_metrics(
    classify_counts: bool = False,
) -> Dict[str, Union[str, List[keras.metrics.Metric]]]:
    """
    Creates the metrics dictionary to use when compiling a model.

    Args:
        classify_counts: Whether we are using the classification count output.

    Returns:
        The metrics dictionary that it created.

    """
    metrics = {}

    if classify_counts:
        # Add a standard accuracy metric for the classification.
        metrics["has_flower"] = [
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.TruePositives(name="tp"),
            keras.metrics.TrueNegatives(name="tn"),
            keras.metrics.FalsePositives(name="fp"),
            keras.metrics.FalseNegatives(name="fn"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ]

    return metrics
