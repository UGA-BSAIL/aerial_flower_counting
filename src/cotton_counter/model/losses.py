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

    Note that for the predictions input, it requires the activation for each
    sub-patch, not for the entire input. It expects this to be formatted as a 2D
    tensor with shape (batch, sub-patch). Also note that it expects the
    sigmoid function to not yet have been applied to the predictions.
    """

    def __init__(
        self, *args: Any, reverse_classes: bool = False, **kwargs: Any,
    ):
        """
        Args:
            *args: Forwarded to superclass constructor.
            reverse_classes: By default, it treats a probability of 0 as
                signifying that an example belongs to the negative class,
                and 1 as signifying that it belongs to the positive class. If
                this is true, however, it will flip those.
            **kwargs: Forwarded to superclass constructor.
        """
        super().__init__(*args, **kwargs)

        # Single-underscore is deliberate in order to avoid autograph issues
        # with name-mangling.
        self._reverse_classes = reverse_classes

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        activation = tf.sigmoid
        if self._reverse_classes:
            # To make the math more intuitive, we always want to treat the
            # output as the probability of the example belonging to the
            # positive class. If the opposite is true, modify the input so
            # that this holds.
            y_true = tf.constant(1, dtype=tf.int32) - y_true

            def activation(x):
                return 1.0 - tf.sigmoid(x)

        # Calculate the probability for the full input.
        input_probability = activation(tf.reduce_mean(y_pred, axis=1))

        # Determine the cross-entropy loss for each sub-patch.
        sub_patch_true = tf.broadcast_to(y_true, tf.shape(y_pred))
        # This will already average across the sub-patch dimension, so there
        # is no need for any further averaging.
        input_losses = tf.keras.losses.binary_crossentropy(
            sub_patch_true, activation(y_pred)
        )

        # Weight by the probability of the full input being negative.
        weighted_input_losses = input_losses * (1.0 - input_probability)

        # Only apply the loss to negative examples.
        negative_example_mask = tf.equal(y_true, 0)
        negative_example_mask = tf.cast(negative_example_mask, tf.float32)
        return weighted_input_losses * negative_example_mask


class PassThroughLoss(losses.Loss):
    """
    This loss simply outputs the predictions it gets fed as the loss values.
    It is intended as part of a hack to work around the fact that Keras doesn't
    support losses that operate on more than one output. Using this loss,
    we can calculate the output values in the model code, and then simply
    output them here.

    """

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        return y_pred


class FocalLoss(losses.Loss):
    """
    Implements focal loss, as described by Lin et al. (2017).
    """

    _EPSILON = tf.constant(0.0001)
    """
    Small constant value to avoid log(0).
    """

    def __init__(
        self, *, alpha: float, gamma: float, **kwargs: Any,
    ):
        """
        Args:
            alpha: Loss weight parameter for the focal loss.
            gamma: Focal strength parameter for the focal loss.
            positive_loss_weight: Additional weight to give the positive
                component of the loss. This is to help balance the
                preponderance of negative samples.
            **kwargs: Will be forwarded to superclass constructor.

        """
        super().__init__(**kwargs)

        self._alpha = tf.constant(alpha)
        self._gamma = tf.constant(gamma)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        one = tf.constant(1.0)
        y_true = tf.cast(y_true, tf.float32)

        # Figure out which locations are positive and which are negative.
        positive_mask = tf.greater(y_true, 0.5)
        positive_pred = tf.boolean_mask(y_pred, positive_mask)
        negative_pred = one - tf.boolean_mask(y_pred, ~positive_mask)
        tf.print("positive_pred", positive_pred)
        tf.print("negative_pred", negative_pred)
        tf.print("num_positive_pred", tf.shape(positive_pred))
        tf.print("num_negative_pred", tf.shape(negative_pred))
        pred_t = tf.concat([positive_pred, negative_pred], axis=-1)

        # Define the loss weight in the same fashion.
        positive_alpha = tf.broadcast_to(self._alpha, tf.shape(positive_pred))
        negative_alpha = tf.broadcast_to(
            1.0 - self._alpha, tf.shape(negative_pred)
        )
        alpha_t = tf.concat([positive_alpha, negative_alpha], axis=-1)

        # Don't allow it to take the log of 0.
        pred_t = tf.maximum(pred_t, self._EPSILON)

        # Compute the focal loss.
        point_loss = -tf.pow(one - pred_t, self._gamma) * tf.math.log(pred_t)
        tf.print(
            "point_loss",
            point_loss,
            "max_point_loss",
            tf.reduce_max(point_loss),
        )
        return alpha_t * point_loss

    def get_config(self) -> Dict[str, Any]:
        super_config = super().get_config()
        return dict(
            alpha=float(self._alpha.numpy()),
            gamma=float(self._gamma.numpy()),
            **super_config,
        )


def make_losses(
    *, alpha: float, gamma: float
) -> Dict[str, Union[str, losses.Loss]]:
    """
    Creates the loss dictionary to use when compiling a model.

    Args:
        alpha: Alpha parameter to use for focal loss.
        gamma: Gamma parameter to use for focal loss.

    Returns:
        The loss dictionary that it created.

    """
    # Use cross-entropy for classification.
    loss_dict = dict(
        has_flower=FocalLoss(alpha=alpha, gamma=gamma, name="pac_loss"),
        combined_bce_loss=PassThroughLoss(name="combined_bce_loss"),
        scale_consistency_loss=PassThroughLoss(name="scale_consistency_loss"),
    )

    return loss_dict
