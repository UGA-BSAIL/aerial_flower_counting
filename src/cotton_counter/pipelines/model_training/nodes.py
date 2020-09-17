"""
Defines nodes for the `model_training` pipeline.
"""

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from loguru import logger

from .model.load_from_dataset import extract_model_input
from .model.losses import CountAccuracy, SparseMse
from .model.sa_net import build_model


def train_model(
    training_data: tf.data.Dataset,
    testing_data: tf.data.Dataset,
    *,
    input_width: int,
    input_height: int,
    patch_scale: float,
    map_height: int,
    map_width: int,
    sigma: int,
    batch_size: int,
    num_prefetch_batches: int,
    learning_rate: float,
    momentum: float
) -> None:
    """
    Trains the model.

    Args:
        training_data: The training dataset to use.
        testing_data: The testing dataset to use.
        input_width: The expected width of the input images, in px.
        input_height: The expected height of the input images, in px.
        patch_scale: The scale factor to apply for the patches we extract.
        map_height: The height of the density maps to create, in px.
        map_width: The width of the density maps to created, ix px.
        sigma:
            The standard deviation in pixels to use for the applied gaussian
            filter.
        batch_size: The size of the batches that we generate.
        num_prefetch_batches: The number of batches to prefetch into memory.
            Increasing this can increase performance at the expense of memory
            usage.
        learning_rate: Learning rate to use for training.
        momentum: Momentum to use for training.

    """
    extraction_kwargs = dict(
        map_shape=(map_height, map_width),
        sigma=sigma,
        batch_size=batch_size,
        num_prefetch_batches=num_prefetch_batches,
        patch_scale=patch_scale,
    )
    training_input = extract_model_input(training_data, **extraction_kwargs)
    testing_input = extract_model_input(testing_data, **extraction_kwargs)

    # Build the model.
    patch_width = int(input_width * patch_scale)
    patch_height = int(input_height * patch_scale)
    model = build_model(input_size=(patch_width, patch_height))
    model.run_eagerly = True
    logger.info("Model has {} parameters.", model.count_params())

    optimizer = keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=momentum
    )
    model.compile(
        optimizer=optimizer,
        loss={"density_map": SparseMse(), "count": CountAccuracy()},
        loss_weights={"density_map": 1.0, "count": 0.0},
    )
    model.fit(
        training_input, validation_data=testing_input, epochs=3,
    )

    optimizer = keras.optimizers.SGD(
        learning_rate=learning_rate * 0.01, momentum=momentum
    )
    model.compile(
        optimizer=optimizer,
        loss=[SparseMse(), CountAccuracy()],
        loss_weights=[1.0, 0.1],
    )
    model.fit(
        training_input,
        validation_data=testing_input,
        batch_size=batch_size,
        validation_batch_size=batch_size,
        epochs=4,
    )
