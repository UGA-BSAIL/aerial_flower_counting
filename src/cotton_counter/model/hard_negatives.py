"""
Utilities related to hard negative mining.
"""


import enum
from typing import Any, Tuple

import numpy as np
import tensorflow as tf
from loguru import logger


@enum.unique
class ThresholdType(enum.IntEnum):
    """
    Represents which type of threshold to use for finding hard
    examples.
    """

    HIGH_SCORE = enum.auto()
    """
    Consider examples with high activations to be hard.
    """
    LOW_SCORE = enum.auto()
    """
    Consider examples with low activations to be hard.
    """


_FILTER_FUNCTIONS = {
    ThresholdType.HIGH_SCORE: lambda x, threshold: x >= threshold,
    ThresholdType.LOW_SCORE: lambda x, threshold: x <= threshold,
}
"""
Filter functions to use for the different types of thresholds.
"""


def _find_activation_threshold(
    *,
    negative_patches: tf.data.Dataset,
    model: tf.keras.Model,
    num_to_keep: int,
    threshold_type: ThresholdType = ThresholdType.HIGH_SCORE,
    batch_size: int = 16,
) -> Tuple[float, np.ndarray]:
    """
    Finds the activation cutoff to use for hard negative mining. Every
    negative instance with an activation below this threshold should be
    dropped from the dataset to keep only the hardest instances.

    Args:
        negative_patches: The dataset containing negative examples.
        model: The model we are training.
        num_to_keep: The total number of negative examples we want to use for
            training.
        threshold_type: Type of thresholding to use for determining which
            examples are hard.
        batch_size: The batch size to use for prediction.

    Returns:
        The activation threshold, as well as an array of the raw activations, in
        the same order as the original dataset.

    """
    # Get all the predictions from the model.
    logger.info("Getting predictions for negative examples...")
    negative_patches = negative_patches.batch(batch_size)
    predictions = model.predict(negative_patches)["discrete_count"].squeeze()

    # Find the ones we did worst on.
    sorted_predictions = np.sort(predictions)
    if threshold_type == ThresholdType.HIGH_SCORE:
        worst_predictions = sorted_predictions[-num_to_keep:]
        activation_threshold = float(worst_predictions[0])
    else:
        worst_predictions = sorted_predictions[:num_to_keep]
        activation_threshold = float(worst_predictions[-1])
    logger.debug("Using activation threshold: {}", activation_threshold)

    return activation_threshold, predictions


def filter_hard_negatives(
    *,
    negative_patches: tf.data.Dataset,
    threshold_type: ThresholdType = ThresholdType.HIGH_SCORE,
    num_to_keep: int,
    **kwargs: Any,
) -> tf.data.Dataset:
    """
    Filters a dataset of negative examples to only include hard negatives.

    Notes:
        Ensure that the dataset does not contain random elements. It must
        always produce elements in the same order.

    Args:
        negative_patches: The dataset of negative examples. (Technically,
            this can also be used to find hard *positive* examples if the
            correct threshold is specified.) It should not be batched.
        threshold_type: Type of thresholding to use for determining which
            examples are hard.
        num_to_keep: The total number of negative examples we want to use for
            training.
        **kwargs: Will be forwarded to `_find_activation_threshold`.

    Returns:
        The same dataset, but containing only hard examples.

    """
    activation_threshold, predictions = _find_activation_threshold(
        negative_patches=negative_patches,
        threshold_type=threshold_type,
        num_to_keep=num_to_keep,
        **kwargs,
    )

    # Associate each example with its model prediction.
    predictions = tf.data.Dataset.from_tensor_slices(predictions)
    examples_with_scores = tf.data.Dataset.zip((negative_patches, predictions))
    # Filter to only the hard examples.
    filter_func = _FILTER_FUNCTIONS[threshold_type]
    hard_negatives = examples_with_scores.filter(
        lambda _, score: filter_func(score, activation_threshold)
    )
    # Remove the extraneous element.
    hard_negatives = hard_negatives.map(lambda example, _: example)

    # In theory, a model could predict the same value for multiple examples,
    # so we do have to enforce the size manually.
    return hard_negatives.take(num_to_keep)
