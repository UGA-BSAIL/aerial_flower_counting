"""
Custom layers for DenseNets, as described in
https://arxiv.org/pdf/1608.06993.pdf
"""


from typing import Any, Dict, Tuple

import tensorflow as tf
from loguru import logger
from tensorflow.keras import layers


class DenseBlock(layers.Layer):
    """
    Implementation of a Dense block. This contains a series of densely-connected
    layers with the same feature map size.
    """

    def __init__(
        self,
        num_layers: int,
        *,
        growth_rate: int,
        use_bottleneck: bool = True,
        bottleneck_ratio: int = 4,
    ):
        """
        Args:
            num_layers: The total number of layers to have in this dense block.
            growth_rate: The growth rate to use for this dense block.
            use_bottleneck: Whether to use bottleneck layers.
            bottleneck_ratio: If using bottleneck layers, this parameter sets
                how many feature maps each 1x1 bottleneck layer will be able
                to produce. This is a factor that is multiplied by the growth
                rate to get the actual number of feature maps.
        """
        super().__init__()

        # We deliberately don't use double-underscores here to avoid
        # Autograph issues with mangled names.
        self._num_layers = num_layers
        self._growth_rate = growth_rate
        self._use_bottleneck = use_bottleneck
        self._bottleneck_ratio = bottleneck_ratio
        self._num_bottleneck_filters = (
            self._bottleneck_ratio * self._growth_rate
        )

        # We have to separate the layer creation from the application of these
        # layers. In order to facilitate this, we use a sub-model.
        self._model = None

    def _composite_function(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Applies the composite function to inputs.

        Args:
            inputs: The inputs to apply the function to.

        Returns:
            The result of the function application.

        """
        if self._use_bottleneck:
            # Add the bottleneck layer as well.
            normalized_bn = layers.BatchNormalization()(inputs)
            relu_bn = layers.ReLU()(normalized_bn)
            inputs = layers.Conv2D(self._num_bottleneck_filters, 1)(relu_bn)

        normalized = layers.BatchNormalization()(inputs)
        relu = layers.ReLU()(normalized)
        return layers.Conv2D(self._growth_rate, 3, padding="same")(relu)

    def build(self, input_shape: Tuple[int, ...]) -> None:
        # The input shape will include the batch size, which we need to pass
        # separately.
        inputs = layers.Input(shape=input_shape[1:], batch_size=input_shape[0])

        # Create the dense connections.
        next_input = inputs
        previous_output_features = [inputs]
        next_output = next_input
        for _ in range(self._num_layers):
            next_output = self._composite_function(next_input)
            previous_output_features.append(next_output)
            next_input = layers.Concatenate()(previous_output_features)

        # Create the model.
        self._model = tf.keras.Model(inputs=inputs, outputs=next_output)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # Apply the sub-model to the inputs.
        return self._model(inputs)

    def get_config(self) -> Dict[str, Any]:
        return dict(
            num_layers=self._num_layers,
            growth_rate=self._growth_rate,
            use_bottleneck=self._use_bottleneck,
            bottleneck_ratio=self._bottleneck_ratio,
        )


class TransitionLayer(layers.Layer):
    """
    Implementation of a transition layer to be used in a dense network,
    which downsamples the input.
    """

    def __init__(self, compression_factor: float = 0.5):
        """
        Args:
            compression_factor: The compression factor to use. This limits
                the number of output feature maps from the transition layer.
        """
        super().__init__()

        # We deliberately don't use double-underscores here to avoid
        # Autograph issues with mangled names.
        self._compression_factor = compression_factor

        # We have to separate the layer creation from the application of these
        # layers. In order to facilitate this, we use a sub-model.
        self._model = None

    def _get_num_output_filters(self, input_shape: Tuple[int, ...]) -> int:
        """
        Determines the number of output filters to use when given the input
        shape to the layer.

        Args:
            input_shape: The full input shape, with or without the batch size.

        Returns:
            The number of output filters to use.

        """
        # Find the number of input filters.
        num_input_filters = input_shape[-1]

        # The number of output filters is determined by the compression factor.
        num_output_filters = int(num_input_filters * self._compression_factor)
        logger.debug("Using {} output filters.", num_input_filters)

        return num_output_filters

    def build(self, input_shape: Tuple[int, ...]) -> None:
        # The input shape will include the batch size, which we need to pass
        # separately.
        inputs = layers.Input(shape=input_shape[1:], batch_size=input_shape[0])

        normalized = layers.BatchNormalization()(inputs)
        num_filters = self._get_num_output_filters(input_shape)
        compressed = layers.Conv2D(num_filters, 1, activation="relu")(
            normalized
        )
        pooled = layers.MaxPool2D()(compressed)

        # Create the sub-model.
        self._model = tf.keras.Model(inputs=inputs, outputs=pooled)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # Apply the sub-model to the inputs.
        return self._model(inputs)

    def get_config(self) -> Dict[str, Any]:
        return dict(compression_factor=self._compression_factor)
