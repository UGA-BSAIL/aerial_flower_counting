"""
Implements the SaNet model architecture.
"""

from functools import partial

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
from loguru import logger

from ..type_helpers import Vector2I


def build_model(*, input_size: Vector2I) -> keras.Model:
    """
    Creates the SaNet model.

    Args:
        input_size: The size of the input images that will be provided,
            in the form (width, height).

    Returns:
        The model that it created.

    """
    # Assume RGB images.
    input_shape = input_size[::-1] + (3,)
    logger.debug("Creating model with input shape {}.", input_shape)
    image_input = keras.Input(shape=input_shape, name="image")

    # Normalize the images before putting them through the model.
    float_images = K.cast(image_input, K.floatx())
    normalized = layers.Lambda(tf.image.per_image_standardization)(
        float_images
    )

    batch_norm = partial(layers.BatchNormalization, momentum=0.5)

    # Main convolution layers.
    conv1_1 = layers.Conv2D(64, 3, padding="same", activation="relu")(
        normalized
    )
    norm1_1 = batch_norm()(conv1_1)
    conv1_2 = layers.Conv2D(64, 3, padding="same", activation="relu")(norm1_1)
    norm1_2 = batch_norm()(conv1_2)
    pool1 = layers.MaxPool2D()(norm1_2)

    conv2_1 = layers.Conv2D(128, 3, padding="same", activation="relu")(pool1)
    norm2_1 = batch_norm()(conv2_1)
    conv2_2 = layers.Conv2D(128, 3, padding="same", activation="relu")(norm2_1)
    norm2_2 = batch_norm()(conv2_2)
    pool2 = layers.MaxPool2D()(norm2_2)

    conv3_1 = layers.Conv2D(256, 3, padding="same", activation="relu")(pool2)
    norm3_1 = batch_norm()(conv3_1)
    conv3_2 = layers.Conv2D(256, 3, padding="same", activation="relu")(norm3_1)
    norm3_2 = batch_norm()(conv3_2)
    conv3_3 = layers.Conv2D(256, 3, padding="same", activation="relu")(norm3_2)
    norm3_3 = batch_norm()(conv3_3)
    pool3 = layers.MaxPool2D()(norm3_3)

    conv4_1 = layers.Conv2D(512, 3, padding="same", activation="relu")(pool3)
    norm4_1 = batch_norm()(conv4_1)
    conv4_2 = layers.Conv2D(512, 3, padding="same", activation="relu")(norm4_1)
    norm4_2 = batch_norm()(conv4_2)

    # Add the projection layers.
    density_map = layers.Conv2D(1, 1, name="density_map")(norm4_2)

    # Sum everything to predict the total count.
    count = layers.Lambda(
        lambda x: K.expand_dims(K.sum(x, axis=[1, 2, 3])), name="count"
    )(density_map)

    # Create the model.
    model = keras.Model(
        inputs=image_input,
        outputs={"density_map": density_map, "count": count},
    )

    return model
