"""
Specialty layers for the WS-Count architecture.
"""


from typing import Any, Dict, Type

import tensorflow as tf

from ..losses import FocalLoss


class CombinedBceLoss(tf.keras.layers.Layer):
    """
    Computes BCE between the PAC predictions and the count predictions,
    in order to enforce that the count actually tracks the objects we care
    about.

    This is deliberately implemented as a custom layer, as a sort of hack to
    get around Keras' inability to compute losses from two separate model
    outputs. Consequently, we compute the scale-consistency loss within
    the model, and then provide that as input to a dummy loss which simply
    returns the predictions.
    """

    def __init__(
        self,
        bce_loss: tf.keras.losses.Loss = tf.keras.losses.BinaryCrossentropy(),
        **kwargs: Any,
    ):
        """

        Args:
            bce_loss: The loss function to use for computing the underlying
                cross-entropy loss.
            **kwargs: Will be forwarded to the superclass.
        """
        super().__init__(**kwargs)

        self._bce_loss = bce_loss

    def call(self, inputs: Dict[str, tf.Tensor], **kwargs: Any,) -> tf.Tensor:
        """
        Computes BCE between the PAC predictions and the count predictions,
        in order to enforce that the count actually tracks the objects we care
        about.

        This is a sort of hack to get around Keras' inability to compute losses
        from two separate model outputs. To get around this, we compute the
        scale-consistency loss within the model, and then provide that as input
        to a dummy loss which simply returns the predictions.

        Args:
            inputs: The input tensors. Should contain the following members:
                pac_predictions: The predictions from the PAC, at every scale.
                    Each item should be a vector, with a length of
                    batch_size * num_patches.
                count_predictions: The predicted counts, at every scale.
                    Should have the same shape as `pac_predictions`.
            **kwargs: Will be forwarded to superclass method.

        Returns:
            The total computed BCE loss for every item in the batch. Will be
            a vector with a length of the batch size.

        """
        pac_predictions = inputs["pac_predictions"]
        count_predictions = inputs["count_predictions"]

        total_loss = None
        # Tracks the number of patches in each input image at this scale.
        num_patches_per_image = 1
        for pac_prediction, count_prediction in zip(
            pac_predictions, count_predictions
        ):
            # Apply sigmoid to the counts, so they're directly comparable to the
            # PAC outputs.
            count_thresholded = tf.keras.activations.sigmoid(
                count_prediction * 8.0 - 4.0
            )
            # Compute cross-entropy loss.
            scale_loss = self._bce_loss.call(pac_prediction, count_thresholded)

            # Combine the losses for all patches from a given image.
            loss_per_input = tf.reshape(
                scale_loss, (-1, num_patches_per_image)
            )
            loss_per_input = tf.reduce_mean(loss_per_input, axis=1)

            if total_loss is None:
                total_loss = loss_per_input
            else:
                total_loss += loss_per_input

            num_patches_per_image *= 4

        # Compute the total loss for all scales.
        return total_loss

    def get_config(self) -> Dict[str, Any]:
        super_config = super().get_config()
        bce_loss_config = self._bce_loss.get_config()
        return dict(bce_loss_config=bce_loss_config, **super_config)

    @classmethod
    def _from_config_with_loss(
        cls, config: Dict[str, Any], *, loss_type: Type[tf.keras.losses.Loss]
    ) -> "CombinedBceLoss":
        """
        Loads a layer from a configuration with a specific loss type.

        Args:
            config: The configuration to load.
            loss_type: The loss type to use.

        Returns:
            The loaded layer.

        """
        bce_loss = loss_type.from_config(config["bce_loss_config"])
        config.pop("bce_loss_config")
        return cls(bce_loss=bce_loss, **config)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CombinedBceLoss":
        return cls._from_config_with_loss(
            config, loss_type=tf.keras.losses.BinaryCrossentropy
        )


class CombinedFocalLoss(CombinedBceLoss):
    """
    Same as `CombinedBceLoss`, but uses focal loss internally instead of raw
    BCE.
    """

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CombinedFocalLoss":
        return cls._from_config_with_loss(config, loss_type=FocalLoss)
