"""
Implementation of MLPconv layer, according to
https://arxiv.org/pdf/1312.4400.pdf
"""


from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers


class MlpConv(layers.Layer):
    """
    Implementation of MLPconv layer, according to
    https://arxiv.org/pdf/1312.4400.pdf
    """

    LayerPartial = Callable[[], layers.Layer]
    """
    Shorthand for the type of a partial function made from a Keras layer.
    """

    def __init__(
        self,
        num_filters: int,
        *args: Any,
        num_mlp_neurons: Optional[int] = None,
        num_mlp_layers: int = 2,
        **kwargs: Any
    ):
        """
        Args:
            num_filters: The number of output filters that the initial nxn
            convolution should have.
            *args: Will be forwarded to `Conv2D`.
            num_mlp_neurons: The number of filters to use for the 1x1
                convolutions. If not specified, it will be the same as
                num_filters.
            num_mlp_layers: The number of 1x1 layers to use after the initial
                convolution.
            **kwargs: Will be forwarded to `Conv2D`.
        """
        super().__init__()

        # We don't handle not specifying keyword arguments explicitly.
        if len(args) > 1:
            raise NameError("Keyword arguments must be specified explicitly.")

        # We deliberately don't use double-underscores here to avoid
        # Autograph issues with mangled names.
        self._conv_args = args
        self._conv_kwargs = kwargs

        self._num_filters = num_filters
        self._num_mlp_neurons = num_filters
        if num_mlp_neurons is not None:
            # Use custom number of neurons for 1x1 layers.
            self._num_mlp_neurons = num_mlp_neurons
        self._num_mlp_layers = num_mlp_layers

        # Create callables for creating the sub-layers.
        self._initial_conv = self._make_initial_conv_partial()
        self._mlp_conv = self._make_mlp_conv_partial()

        # List of the sub-layers that we use.
        self._sub_layers = []

    def _make_initial_conv_partial(self) -> LayerPartial:
        """
        Creates the partial function that we can use for creating the main
        convolutional layer.

        Returns:
            The partial function that it created.

        """
        return partial(
            layers.Conv2D,
            self._num_filters,
            *self._conv_args,
            **self._conv_kwargs,
        )

    def _make_mlp_conv_partial(self) -> LayerPartial:
        """
        Creates the partial function that we can use for creating the subsequent
        1x1 convolutional layers.

        Returns:
            The partial function that it created.

        """
        # There are certain arguments that shouldn't be propagated to these
        # layers.
        kwargs = self._conv_kwargs.copy()
        for invalid_key in ("strides", "padding", "dilation_rate"):
            kwargs.pop(invalid_key, None)

        return partial(
            layers.Conv2D, self._num_mlp_neurons, kernel_size=1, **kwargs,
        )

    def build(self, input_shape: Tuple[int, ...]) -> None:
        super().build(input_shape)

        # Create the sub-layers.
        self._sub_layers.append(self._initial_conv())
        for _ in range(self._num_mlp_layers):
            self._sub_layers.append(self._mlp_conv())

    def call(self, inputs: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        # Apply the sub-layers.
        head = inputs
        for layer in self._sub_layers:
            head = layer(head)

        return head

    def get_config(self) -> Dict[str, Any]:
        return dict(
            num_filters=self._num_filters,
            conv_args=self._conv_args,
            conv_kwargs=self._conv_kwargs,
            num_mlp_neurons=self._num_mlp_neurons,
            num_mlp_layers=self._num_mlp_layers,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MlpConv":
        num_filters = config.pop("num_filters")
        args = config.pop("conv_args")
        kwargs = config.pop("conv_kwargs")

        # Everything else we should be able to pass straight through.
        return cls(num_filters, *args, **config, **kwargs)
