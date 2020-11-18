"""
Custom Keras losses.
"""


from typing import Any, Dict, Union

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


class ScaleConsistentBinaryCrossEntropy(losses.Loss):
    """
    A version of binary cross-entropy loss that can be applied to sub-divided
    activations of the model.

    The purpose of this loss is to penalize cases where the model predicts the
    negative class overall, but still predicts the positive class for some
    sub-patches.

    Note that for the predictions input, it requires the probability for each
    sub-patch, not for the entire input. It expects this to be formatted as a 2D
    tensor with shape (batch, sub-patch)
    """

    def __init__(
        self,
        *args: Any,
        reverse_classes: bool = False,
        threshold: float = 0.5,
        **kwargs: Any
    ):
        """
        Args:
            *args: Forwarded to superclass constructor.
            reverse_classes: By default, it treats a probability of 0 as
                signifying that an example belongs to the negative class,
                and 1 as signifying that it belongs to the positive class. If
                this is true, however, it will flip those.
            threshold: If the overall probability that the input example belongs
                to the positive class is above this threshold, the value of this
                loss will be zero. This is used to prevent the loss from
                unfairly penalizing positive examples.
            **kwargs: Forwarded to superclass constructor.
        """
        super().__init__(*args, **kwargs)

        # Single-underscore is deliberate in order to avoid autograph issues
        # with name-mangling.
        self._reverse_classes = reverse_classes
        self._threshold = threshold

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        if self._reverse_classes:
            # To make the math more intuitive, we always want to treat the
            # output as the probability of the example belonging to the
            # positive class. If the opposite is true, modify the input so
            # that this holds.
            y_true = tf.constant(1, dtype=tf.int32) - y_true
            y_pred = tf.constant(1.0, dtype=tf.float32) - y_pred

        # Calculate the probability for the full input.
        input_probability = tf.reduce_mean(y_pred, axis=1)

        # Determine the cross-entropy loss for each sub-patch.
        sub_patch_true = tf.broadcast_to(y_true, tf.shape(y_pred))
        # This will already average across the sub-patch dimension, so there
        # is no need for any further averaging.
        input_losses = tf.keras.losses.binary_crossentropy(
            sub_patch_true, y_pred
        )

        # Weight by the probability of the full input being negative.
        weighted_input_losses = input_losses * (1.0 - input_probability)

        # Obey the threshold, zeroing anything that falls above it.
        threshold_mask = input_probability <= self._threshold
        threshold_mask = tf.cast(threshold_mask, tf.float32)
        return weighted_input_losses * threshold_mask


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
        loss_dict[
            "discrete_sub_patch_count"
        ] = ScaleConsistentBinaryCrossEntropy(reverse_classes=True)

    return loss_dict
