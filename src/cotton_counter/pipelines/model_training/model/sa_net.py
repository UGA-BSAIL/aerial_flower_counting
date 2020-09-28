"""
Implements the SaNet model architecture.
"""

from typing import Optional

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
from loguru import logger

from ..type_helpers import Vector2I


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
    return keras.Input(shape=input_shape, name="image")


def _build_model_backbone(*, image_input: keras.Input) -> layers.Layer:
    """
    Creates the backbone SaNet model.

    Args:
        image_input: The image input to build the model off of.

    Returns:
        The top model layer.

    """
    # Normalize the images before putting them through the model.
    float_images = K.cast(image_input, K.floatx())
    normalized = layers.Lambda(tf.image.per_image_standardization)(
        float_images
    )

    # Main convolution layers.
    conv1_1 = layers.Conv2D(48, 3, padding="same", activation="relu")(
        normalized
    )
    conv1_2 = layers.Conv2D(48, 3, padding="same", activation="relu")(conv1_1)
    pool1 = layers.MaxPool2D()(conv1_2)

    conv2_1 = layers.Conv2D(96, 3, padding="same", activation="relu")(pool1)
    conv2_2 = layers.Conv2D(96, 1, activation="relu")(conv2_1)
    conv2_3 = layers.Conv2D(96, 1, activation="relu")(conv2_2)
    conv2_4 = layers.Conv2D(96, 3, padding="same", activation="relu")(conv2_3)
    conv2_5 = layers.Conv2D(96, 1, activation="relu")(conv2_4)
    conv2_6 = layers.Conv2D(96, 1, activation="relu")(conv2_5)
    pool2 = layers.MaxPool2D()(conv2_6)

    conv3_1 = layers.Conv2D(192, 3, padding="same", activation="relu")(pool2)
    conv3_2 = layers.Conv2D(192, 1, padding="same", activation="relu")(conv3_1)
    conv3_3 = layers.Conv2D(192, 1, padding="same", activation="relu")(conv3_2)
    conv3_4 = layers.Conv2D(192, 3, padding="same", activation="relu")(conv3_3)
    conv3_5 = layers.Conv2D(192, 1, padding="same", activation="relu")(conv3_4)
    conv3_6 = layers.Conv2D(192, 1, padding="same", activation="relu")(conv3_5)
    conv3_7 = layers.Conv2D(192, 3, padding="same", activation="relu")(conv3_6)
    conv3_8 = layers.Conv2D(192, 1, padding="same", activation="relu")(conv3_7)
    conv3_9 = layers.Conv2D(192, 1, padding="same", activation="relu")(conv3_8)
    pool3 = layers.MaxPool2D()(conv3_9)

    conv4_1 = layers.Conv2D(384, 3, padding="same", activation="relu")(pool3)
    conv4_2 = layers.Conv2D(384, 1, activation="relu")(conv4_1)
    conv4_3 = layers.Conv2D(384, 1, activation="relu")(conv4_2)
    conv4_4 = layers.Conv2D(384, 3, padding="same", activation="relu")(conv4_3)
    conv4_5 = layers.Conv2D(128, 1, activation="relu")(conv4_4)
    conv4_6 = layers.Conv2D(128, 1, activation="relu")(conv4_5)

    return conv4_6


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


def _build_count_classification_head(
    model_top: layers.Layer, *, num_classes: int
) -> layers.Layer:
    """
    Adds the head for classifying categorical count values.

    Args:
        model_top: The top model layer to build the head on.
        num_classes: The total number of count classes we have.

    Returns:
        The layer representing the categorical count logits.

    """
    count_conv_1 = layers.Conv2D(num_classes, 1, activation="relu")(model_top)
    count_pool_1 = layers.GlobalAveragePooling2D()(count_conv_1)
    count_softmax = layers.Softmax()(count_pool_1)

    return count_softmax


def build_model(
    *, input_size: Vector2I, num_classes: Optional[int] = None
) -> keras.Model:
    """
    Creates the full SaNet model.

    Args:
        input_size: The size of the input images that will be provided,
            in the form (width, height).
        num_classes: The number of classes to use if we want to use the count
            classification output. If this is `None`, it will assume we want to
            regress the counts directly.

    Returns:
        The model that it created.

    """
    image_input = _build_image_input(input_size=input_size)
    backbone = _build_model_backbone(image_input=image_input)

    density_map = _build_density_map_head(backbone)
    model_outputs = {"density_map": density_map}

    # Build the regression head.
    count = _build_count_regression_head(density_head=density_map)
    model_outputs["count"] = count

    if num_classes is not None:
        # Use the classification head.
        discrete_count = _build_count_classification_head(
            backbone, num_classes=num_classes
        )
        model_outputs["discrete_count"] = discrete_count

    # Create the model.
    model = keras.Model(inputs=image_input, outputs=model_outputs,)

    return model
