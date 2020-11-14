"""
Implements the model architecture.
"""

from typing import Optional

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
from loguru import logger

from src.cotton_counter.type_helpers import Vector2I

from ..model.layers import DenseBlock, TransitionLayer


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
    dense1 = DenseBlock(6, growth_rate=12)(pool1)
    transition1 = TransitionLayer()(dense1)

    dense2 = DenseBlock(12, growth_rate=12)(transition1)
    transition2 = TransitionLayer()(dense2)

    dense3 = DenseBlock(24, growth_rate=12)(transition2)
    transition3 = TransitionLayer()(dense3)

    dense4 = DenseBlock(16, growth_rate=12)(transition3)
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


def _build_count_classification_head(model_top: layers.Layer) -> layers.Layer:
    """
    Adds the head for classifying categorical count values.

    Args:
        model_top: The top model layer to build the head on.

    Returns:
        The layer representing the categorical count logits.

    """
    count_conv_1 = layers.Conv2D(1, 1, name="activation_maps")(model_top)
    count_pool_1 = layers.GlobalAveragePooling2D()(count_conv_1)
    count_sigmoid = layers.Activation("sigmoid", name="discrete_count")(
        count_pool_1
    )

    return count_sigmoid


def build_model(*, input_size: Vector2I, use_mil: bool = False) -> keras.Model:
    """
    Creates the full SaNet model.

    Args:
        input_size: The size of the input images that will be provided,
            in the form (width, height).
        use_mil: Whether to use a binary output for the MIL task. Otherwise,
            it will assume that we want to regress the counts directly.

    Returns:
        The model that it created.

    """
    image_input = _build_image_input(input_size=input_size)
    backbone = _build_model_backbone(image_input=image_input)

    model_outputs = {}
    if use_mil:
        # Use the classification head.
        discrete_count = _build_count_classification_head(backbone)
        model_outputs["discrete_count"] = discrete_count

    # Create the model.
    model = keras.Model(inputs=image_input, outputs=model_outputs,)

    return model
