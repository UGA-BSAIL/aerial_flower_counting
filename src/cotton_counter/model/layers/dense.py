"""
Custom layers for DenseNets, as described in
https://arxiv.org/pdf/1608.06993.pdf
"""


from functools import partial
from typing import Any, Dict, Tuple

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

        # Weights for the bottleneck batchnorm.
        self._bottleneck_gamma = None
        self._bottleneck_beta = None
        # Weights for the bottleneck convolution.
        self._bottleneck_conv_kernel = None
        self._bottleneck_conv_bias = None
        # Weights for the standard batchnorm.
        self._gamma = None
        self._beta = None
        # Weights for the standard convolution.
        self._conv_kernel = None
        self._conv_bias = None

    @staticmethod
    def _batch_norm(
        inputs: tf.Tensor, *, gamma: tf.Tensor, beta: tf.Tensor
    ) -> tf.Tensor:
        """
        Performs batch normalization on an input. Note that, unlike the Keras
        layer, this does not do any fancy moving averages.

        Args:
            inputs: The inputs to normalize.
            gamma: The learnable scale.
            beta: The learnable offset.

        Returns:
            The normalized outputs.

        """
        mean, variance = tf.nn.moments(inputs, axes=[0])
        return tf.nn.batch_normalization(
            inputs,
            mean,
            variance,
            offset=beta,
            scale=gamma,
            variance_epsilon=0.0001,
        )

    def _add_weights(self, input_shape: Tuple[int, ...]) -> None:
        """
        Creates the necessary weights for this layer.

        Args:
            input_shape: The shape of the layer input.

        """
        _, input_height, input_width, input_channels = input_shape
        normal_conv_input_filters = input_channels

        if self._use_bottleneck:
            # Add the bottleneck batchnorm weights.
            self._bottleneck_gamma = self.add_weight(
                name="bottleneck_gamma", shape=(input_channels,)
            )
            self._bottleneck_beta = self.add_weight(
                name="bottleneck_beta", shape=(input_channels,)
            )

            # Add the bottleneck convolution weights.
            self._bottleneck_conv_kernel = self.add_weight(
                name="bottleneck_conv_kernel",
                shape=(1, 1, input_channels, self._num_bottleneck_filters),
            )
            self._bottleneck_conv_bias = self.add_weight(
                name="bottleneck_conv_bias",
                shape=(self._num_bottleneck_filters,),
            )

            # Determine number of input filters for the normal convolution.
            normal_conv_input_filters = self._num_bottleneck_filters

        # Add the standard batchnorm weights.
        self._gamma = self.add_weight(
            name="gamma", shape=(normal_conv_input_filters,)
        )
        self._beta = self.add_weight(
            name="beta", shape=(normal_conv_input_filters,)
        )

        # Add the standard convolution weights.
        self._conv_kernel = self.add_weight(
            name="conv_kernel",
            shape=(3, 3, normal_conv_input_filters, self._growth_rate),
        )
        self._conv_bias = self.add_weight(
            name="conv_bias", shape=(self._growth_rate,)
        )

    def build(self, input_shape: Tuple[int, ...]) -> None:
        self._add_weights(input_shape)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, **kwargs: Any):
        def _apply_layer(_inputs: tf.Tensor) -> tf.Tensor:
            # Add the layer operations.
            if self._use_bottleneck:
                # Add the bottleneck layer as well.
                normalized_bn = self._batch_norm(
                    _inputs,
                    gamma=self._bottleneck_gamma,
                    beta=self._bottleneck_beta,
                )
                relu_bn = tf.nn.relu(normalized_bn)
                conv_bn = tf.nn.conv2d(
                    relu_bn,
                    self._bottleneck_conv_kernel,
                    strides=1,
                    padding="SAME",
                )
                _inputs = tf.nn.bias_add(conv_bn, self._bottleneck_conv_bias)

            normalized = self._batch_norm(
                _inputs, gamma=self._gamma, beta=self._beta
            )
            relu = tf.nn.relu(normalized)
            conv = tf.nn.conv2d(
                relu, self._conv_kernel, strides=1, padding="SAME"
            )
            return tf.nn.bias_add(conv, self._conv_bias)

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
        self._composite_function = partial(_CompositeFunction, **kwargs)

        # We have to separate the layer creation from the application of these
        # layers. In order to facilitate this, we use a sub-model.
        self._model = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        # The input shape will include the batch size, which we need to pass
        # separately.
        inputs = layers.Input(shape=input_shape[1:], batch_size=input_shape[0])

        # Create the dense connections.
        next_input = inputs
        previous_output_features = [inputs]
        next_output = next_input
        for _ in range(self._num_layers):
            next_output = self._composite_function()(next_input)
            previous_output_features.append(next_output)
            next_input = layers.Concatenate()(previous_output_features)

        # Create the model.
        self._model = tf.keras.Model(inputs=inputs, outputs=next_output)

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        # Apply the sub-model to the inputs.
        return self._model(inputs)

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

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # Apply the sub-model to the inputs.
        return self._model(inputs)

    def get_config(self) -> Dict[str, Any]:
        return dict(compression_factor=self._compression_factor)
