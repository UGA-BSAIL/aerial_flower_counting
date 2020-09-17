"""
Custom Keras losses.
"""


import tensorflow as tf
import tensorflow.keras.losses as losses


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

        zero_loss = tf.square((predicted_zero - true_zero) / (true_zero + 1))
        zero_loss = tf.reduce_mean(zero_loss)
        non_zero_loss = tf.square(
            (predicted_non_zero - true_non_zero) / (true_non_zero + 1)
        )
        non_zero_loss = tf.reduce_mean(non_zero_loss)
        tf.print(true_count, predicted_count, zero_loss, non_zero_loss)
        return non_zero_loss + 0.1 * zero_loss


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

        zero_loss = tf.square(predicted_zero - true_zero)
        zero_loss = tf.reduce_mean(zero_loss)
        non_zero_loss = tf.square(predicted_non_zero - true_non_zero)
        non_zero_loss = tf.reduce_mean(non_zero_loss)

        return non_zero_loss + zero_loss * 0.1
