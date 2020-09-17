"""
Implements the SaNet model architecture.
"""

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

    # Normalize the image.
    normalized = K.cast(image_input, K.floatx())
    normalized /= 255.0

    # Main convolution layers.
    conv1_1 = layers.Conv2D(64, 3, padding="same", activation="relu")(
        normalized
    )
    conv1_2 = layers.Conv2D(64, 3, padding="same", activation="relu")(conv1_1)
    pool1 = layers.MaxPool2D()(conv1_2)

    conv2_1 = layers.Conv2D(128, 3, padding="same", activation="relu")(pool1)
    conv2_2 = layers.Conv2D(128, 3, padding="same", activation="relu")(conv2_1)
    pool2 = layers.MaxPool2D()(conv2_2)

    conv3_1 = layers.Conv2D(256, 3, padding="same", activation="relu")(pool2)
    conv3_2 = layers.Conv2D(256, 3, padding="same", activation="relu")(conv3_1)
    conv3_3 = layers.Conv2D(256, 3, padding="same", activation="relu")(conv3_2)
    pool3 = layers.MaxPool2D()(conv3_3)

    conv4_1 = layers.Conv2D(512, 3, padding="same", activation="relu")(pool3)
    conv4_2 = layers.Conv2D(512, 3, padding="same", activation="relu")(conv4_1)

    # Add the projection layers.
    p_conv3 = layers.Conv2D(1, 1)(conv4_2)
    # Normalize between 1 and 0.
    density_map = layers.Lambda(
        lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)), name="density_map"
    )(p_conv3)

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
