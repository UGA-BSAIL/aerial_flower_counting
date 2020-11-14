"""
Custom Keras losses.
"""


from typing import Dict, Union

import tensorflow as tf
import tensorflow.keras.losses as losses


def _mse(
    true: tf.Tensor, predicted: tf.Tensor, normalize: bool = False
) -> tf.Tensor:
    """
    A normalized version of MSE.

    Args:
        true: The true values.
        predicted: The predicted values.
        normalize: If true, the squares will be divided by the true values
            before taking the mean.

    Returns:
        The normalized MSE value for the entire batch.

    """

    def _do_mse() -> tf.Tensor:
        delta = predicted - true
        if normalize:
            delta /= true + tf.constant(1.0, dtype=tf.float32)

        loss = tf.square(delta)
        return tf.reduce_mean(loss)

    # A corner case is if we have no data. In that situation, we want the
    # MSE to be 0, since that's what makes the most sense for a loss.
    return tf.cond(
        tf.equal(tf.size(true), 0),
        lambda: tf.constant(0.0, tf.float32),
        _do_mse,
    )


class CountAccuracy(losses.Loss):
    """
    A loss that computes that accuracy of the predicted counts.
    """

    def call(
        self, true_count: tf.Tensor, predicted_count: tf.Tensor
    ) -> tf.Tensor:
        true_non_zero_mask = true_count > tf.constant(0.001)
        true_non_zero = true_count[true_non_zero_mask]
        predicted_non_zero = predicted_count[true_non_zero_mask]

        true_zero = true_count[~true_non_zero_mask]
        predicted_zero = predicted_count[~true_non_zero_mask]

        zero_loss = _mse(true_zero, predicted_zero, normalize=True)
        non_zero_loss = _mse(true_non_zero, predicted_non_zero, normalize=True)

        return non_zero_loss + tf.constant(0.1) * zero_loss


class SparseMse(losses.Loss):
    """
    A version of MSE that weights zero pixels less than non-zero pixels.
    """

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # Find density maps with zero counts.
        true_count = tf.reduce_sum(y_true, axis=[1, 2, 3])
        true_non_zero_mask = true_count > 0.001

        true_non_zero = y_true[true_non_zero_mask]
        predicted_non_zero = y_pred[true_non_zero_mask]
        true_zero = y_true[~true_non_zero_mask]
        predicted_zero = y_pred[~true_non_zero_mask]

        zero_loss = _mse(true_zero, predicted_zero)
        non_zero_loss = _mse(true_non_zero, predicted_non_zero)

        return non_zero_loss + zero_loss * tf.constant(1.0)


def make_losses(
    classify_counts: bool = False,
) -> Dict[str, Union[str, losses.Loss]]:
    """
    Creates the loss dictionary to use when compiling a model.

    Args:
        classify_counts: Whether we are using the classification count output.

    Returns:
        The loss dictionary that it created.

    """
    loss_dict = {}
    if classify_counts:
        # Use cross-entropy for classification.
        loss_dict["discrete_count"] = "binary_crossentropy"

    return loss_dict
