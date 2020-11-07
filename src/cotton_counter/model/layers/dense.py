"""
Custom layers for DenseNets, as described in
https://arxiv.org/pdf/1608.06993.pdf
"""


from functools import partial
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf
from loguru import logger
from tensorflow.keras import layers


class _CompositeFunction(layers.Layer):
    """
    We break the composite function out into its own layer. This is because
    gradient checkpointing currently doesn't work correctly with Keras layers,
    so we have to implement everything here manually.
    """

    def __init__(
        self,
        *,
        growth_rate: int,
        use_bottleneck: bool = True,
        bottleneck_ratio: int = 4,
        reduce_memory: bool = True,
    ):
        """
        Args:
            growth_rate: The growth rate to use for this dense block.
            use_bottleneck: Whether to use bottleneck layers.
            bottleneck_ratio: If using bottleneck layers, this parameter sets
                how many feature maps each 1x1 bottleneck layer will be able
                to produce. This is a factor that is multiplied by the growth
                rate to get the actual number of feature maps.
            reduce_memory: If true, it will use gradient checkpointing to
                reduce memory usage at the expense of slightly more compute
                time.
        """
        super().__init__()

        self._growth_rate = growth_rate
        self._use_bottleneck = use_bottleneck
        self._bottleneck_ratio = bottleneck_ratio
        self._num_bottleneck_filters = (
            self._bottleneck_ratio * self._growth_rate
        )
        self._reduce_memory = reduce_memory

        # We initialize layers here, because it turns out that using the usual
        # sub-model strategy breaks recompute_grad in exciting ways.
        # Batch normalization layers.
        self._bottleneck_norm = layers.BatchNormalization()
        self._norm = layers.BatchNormalization()
        # Convolutional layers.
        self._bottleneck_conv = layers.Conv2D(self._num_bottleneck_filters, 1)
        self._conv = layers.Conv2D(self._growth_rate, 3, padding="same")

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None):
        def _apply_layer(_inputs: tf.Tensor) -> tf.Tensor:
            # Add the layer operations.
            if self._use_bottleneck:
                # Add the bottleneck layer as well.
                normalized_bn = self._bottleneck_norm(
                    _inputs, training=training
                )
                relu_bn = layers.ReLU()(normalized_bn)
                _inputs = self._bottleneck_conv(relu_bn)

            normalized = self._norm(_inputs, training=training)
            relu = layers.ReLU()(normalized)
            return self._conv(relu)

        if self._reduce_memory:
            # Force gradient checkpointing.
            _apply_layer = tf.recompute_grad(_apply_layer)
        return _apply_layer(inputs)

    def get_config(self) -> Dict[str, Any]:
        return dict(
            growth_rate=self._growth_rate,
            use_bottleneck=self._use_bottleneck,
            bottleneck_ratio=self._bottleneck_ratio,
        )


class DenseBlock(layers.Layer):
    """
    Implementation of a Dense block. This contains a series of densely-connected
    layers with the same feature map size.
    """

    def __init__(self, num_layers: int, **kwargs: Any):
        """
        Args:
            num_layers: The total number of layers to have in this dense block.
            **kwargs: Additional arguments will be forwarded to the
                constructor of `_CompositeFunction`.
        """
        super().__init__()

        # We deliberately don't use double-underscores here to avoid
        # Autograph issues with mangled names.
        self._num_layers = num_layers
        self._kwargs = kwargs

        # Pre-create the composite function layers.
        composite_function = partial(_CompositeFunction, **kwargs)
        self._composite_function_layers = [
            composite_function() for _ in range(num_layers)
        ]

    def call(
        self, inputs: tf.Tensor, training: Optional[bool] = None
    ) -> tf.Tensor:
        # Create the dense connections.
        next_input = inputs
        previous_output_features = [inputs]
        next_output = next_input
        for composite_function_layer in self._composite_function_layers:
            next_output = composite_function_layer(
                next_input, training=training
            )
            previous_output_features.append(next_output)
            next_input = layers.Concatenate()(previous_output_features)

        return next_output

    def get_config(self) -> Dict[str, Any]:
        return dict(num_layers=self._num_layers, **self._kwargs)


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

        # Pre-create the sub-layers.
        self._norm = layers.BatchNormalization()
        # Can't be initialized until we know the input shape.
        self._conv = None

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
        # Calculate the number of filters and use it to initialize the
        # convolution layer.
        num_filters = self._get_num_output_filters(input_shape)
        self._conv = layers.Conv2D(num_filters, 1, activation="relu")

        super().build(input_shape)

    def call(
        self, inputs: tf.Tensor, training: Optional[bool] = None
    ) -> tf.Tensor:
        normalized = self._norm(inputs, training=training)
        compressed = self._conv(normalized)
        return layers.MaxPool2D()(compressed)

    def get_config(self) -> Dict[str, Any]:
        return dict(compression_factor=self._compression_factor)
