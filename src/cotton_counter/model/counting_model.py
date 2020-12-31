"""
Implements the model architecture.
"""

from typing import Optional, Tuple

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
from loguru import logger

from src.cotton_counter.type_helpers import Vector2I

from ..model.layers import DenseBlock, TransitionLayer
from ..model.patches import extract_standard_patches


def _build_image_input(*, input_size: Vector2I) -> keras.Input:
    """
    Creates the image data input.

    Args:
        input_size: The size of the input images that will be provided,
            in the form (width, height).

    Returns:
        The `Input` that it created.

    """
    # Assume RGB images.
    input_shape = input_size[::-1] + (3,)
    logger.debug("Creating model with input shape {}.", input_shape)
    return keras.Input(shape=input_shape, name="image", dtype="uint8")


def _build_model_backbone(*, image_input: keras.Input) -> layers.Layer:
    """
    Creates the backbone SaNet model.

    Args:
        image_input: The image input to build the model off of.

    Returns:
        The top model layer.

    """
    # Normalize the images before putting them through the model.
    float_images = tf.cast(image_input, K.floatx())
    normalized = tf.image.per_image_standardization(float_images)

    # Input convolution layers.
    conv1_1 = layers.Conv2D(48, 3, padding="same", activation="relu")(
        normalized
    )
    norm1_1 = layers.BatchNormalization()(conv1_1)
    conv1_2 = layers.Conv2D(48, 3, padding="same", activation="relu")(norm1_1)
    # No normalization needed here because dense blocks normalize internally.
    pool1 = layers.MaxPool2D()(conv1_2)

    # Dense blocks.
    dense1 = DenseBlock(3, growth_rate=4)(pool1)
    transition1 = TransitionLayer()(dense1)

    dense2 = DenseBlock(6, growth_rate=4)(transition1)
    transition2 = TransitionLayer()(dense2)

    dense3 = DenseBlock(12, growth_rate=4)(transition2)
    transition3 = TransitionLayer()(dense3)

    dense4 = DenseBlock(8, growth_rate=4)(transition3)
    transition4 = TransitionLayer()(dense4)

    return transition4


def _build_density_map_head(model_top: layers.Layer) -> layers.Layer:
    """
    Adds the head for predicting density maps.

    Args:
        model_top: The top model layer to build the head on.

    Returns:
        The layer representing the density map output.

    """
    return layers.Conv2D(1, 1, name="density_map")(model_top)


def _build_count_regression_head(
    *, density_head: layers.Layer
) -> layers.Layer:
    """
    Adds the head for regressing count values.

    Args:
        density_head: The layer that produces the density map output.

    Returns:
        The layer representing the count output.

    """
    # Sum everything to predict the total count.
    return layers.Lambda(
        lambda x: K.expand_dims(K.sum(x, axis=[1, 2, 3])), name="count"
    )(density_head)


def _apply_sub_patch_classification(
    activation_maps: tf.Tensor,
    *,
    sub_patch_scale: float,
    sub_patch_stride: Optional[float] = None
) -> tf.Tensor:
    """
    Extracts sub-patches from a batch of activation maps and computes the
    average activation for each sub-patch.

    Args:
        activation_maps: The batch of activation maps.
        sub_patch_scale: The scale of the sub-patches to extract.
        sub_patch_stride: The stride of the sub-patches to extract. If not
            specified, it will extract non-overlapping sub-patches.

    Returns:
        The sub-patch activations. It will be a tensor with a shape of
        (batch, sub-patch).

    """
    if sub_patch_stride is None:
        # Default to non-overlapping.
        sub_patch_stride = sub_patch_scale

    # Extract the sub-patches from the activation map.
    sub_patches = extract_standard_patches(
        activation_maps,
        patch_scale=sub_patch_scale,
        patch_stride=sub_patch_stride,
    )

    # Perform global average pooling.
    sub_patches_mean = tf.reduce_mean(sub_patches, axis=[1, 2, 3])

    # Reshape to maintain the original batch size.
    batch_size = tf.shape(activation_maps)[0]
    return tf.reshape(sub_patches_mean, (batch_size, -1))


def _build_count_classification_head(
    model_top: layers.Layer,
    *,
    sub_patch_scale: float,
    sub_patch_stride: float,
    output_bias: Optional[float] = None
) -> Tuple[layers.Layer, layers.Layer]:
    """
    Adds the head for classifying categorical count values.

    Args:
        model_top: The top model layer to build the head on.
        sub_patch_scale: The scale of the sub-patches to extract.
        sub_patch_stride: The stride of the sub-patches to extract. If not
            specified, it will extract non-overlapping sub-patches.
        output_bias: Specify an initial bias to use for the output. This can
            be useful for unbalanced datasets.

    Returns:
        The layer representing the categorical count logits, and the layer
        representing the average activations for each sub-patch.

    """
    if output_bias is not None:
        # Use the specified bias.
        logger.debug("Using initial output bias {}.", output_bias)
        output_bias = keras.initializers.Constant(output_bias)

    count_conv_1 = layers.Conv2D(
        1, 1, name="activation_maps", bias_initializer=output_bias
    )(model_top)

    # Generate the sub-patch output.
    sub_patch_pool_1 = layers.Lambda(
        _apply_sub_patch_classification,
        arguments=dict(
            sub_patch_scale=sub_patch_scale, sub_patch_stride=sub_patch_stride,
        ),
        name="discrete_sub_patch_count",
    )(count_conv_1)
    # We deliberately return the raw values for the sub-patches instead of the
    # sigmoid. This is an implementation hack to get around the fact that Keras
    # does not allow the use of multiple outputs in a single loss function.

    count_pool_1 = layers.GlobalAveragePooling2D()(count_conv_1)
    count_sigmoid = layers.Activation("sigmoid", name="discrete_count")(
        count_pool_1
    )

    return count_sigmoid, sub_patch_pool_1


def build_model(
    *,
    input_size: Vector2I,
    use_mil: bool = False,
    sub_patch_scale: float = 0.5,
    sub_patch_stride: float = 0.5,
    output_bias: Optional[float] = None
) -> keras.Model:
    """
    Creates the full SaNet model.

    Args:
        input_size: The size of the input images that will be provided,
            in the form (width, height).
        use_mil: Whether to use a binary output for the MIL task. Otherwise,
            it will assume that we want to regress the counts directly.
        sub_patch_scale: The scale of the sub-patches to extract.
        sub_patch_stride: The stride of the sub-patches to extract. If not
            specified, it will extract non-overlapping sub-patches.
        output_bias: Specify an initial bias to use for the output. This can
            be useful for unbalanced datasets.

    Returns:
        The model that it created.

    """
    image_input = _build_image_input(input_size=input_size)
    backbone = _build_model_backbone(image_input=image_input)

    model_outputs = {}
    if use_mil:
        # Use the classification head.
        (
            discrete_count,
            discrete_sub_patch_count,
        ) = _build_count_classification_head(
            backbone,
            sub_patch_scale=sub_patch_scale,
            sub_patch_stride=sub_patch_stride,
            output_bias=output_bias,
        )
        model_outputs["discrete_count"] = discrete_count
        model_outputs["discrete_sub_patch_count"] = discrete_sub_patch_count

    # Create the model.
    model = keras.Model(inputs=image_input, outputs=model_outputs,)

    return model
