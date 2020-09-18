"""
Defines nodes for the `model_training` pipeline.
"""

from typing import Any, Dict, List

import tensorflow as tf
import tensorflow.keras as keras
from loguru import logger

from .model.load_from_dataset import extract_model_input
from .model.losses import CountAccuracy, SparseMse
from .model.sa_net import build_model


def pre_process_dataset(
    raw_dataset: tf.data.Dataset,
    *,
    patch_scale: float,
    map_height: int,
    map_width: int,
    sigma: int,
    batch_size: int,
    num_prefetch_batches: int,
) -> tf.data.Dataset:
    """
    Generates the `Datasets` containing pre-processed data to use for
    training the model.

    Args:
        raw_dataset: The `Dataset` containing raw data that needs to be
            converted to a form usable by the model.
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

    Returns:
       A new `Dataset` containing pre-processed data that is ready to use as
       model input.

    """
    extraction_kwargs = dict(
        map_shape=(map_height, map_width),
        sigma=sigma,
        batch_size=batch_size,
        num_prefetch_batches=num_prefetch_batches,
        patch_scale=patch_scale,
    )
    return extract_model_input(raw_dataset, **extraction_kwargs)


def create_model(
    input_width: int, input_height: int, patch_scale: float,
) -> keras.Model:
    """
    Builds the model to use.

    Args:
        input_width: The expected width of the input images, in px.
        input_height: The expected height of the input images, in px.
        patch_scale: The scale factor to apply for the patches we extract.

    Returns:
        The model that it created.

    """
    # Build the model.
    patch_width = int(input_width * patch_scale)
    patch_height = int(input_height * patch_scale)
    model = build_model(input_size=(patch_width, patch_height))
    logger.info("Model has {} parameters.", model.count_params())

    return model


def train_model(
    model: keras.Model,
    *,
    training_data: tf.data.Dataset,
    testing_data: tf.data.Dataset,
    learning_phases: List[Dict[str, Any]],
) -> keras.Model:
    """
    Trains a model.

    Args:
        model: The model to train.
        training_data: The `Dataset` containing pre-processed training data.
        testing_data: The `Dataset` containing pre-processed testing data.
        learning_phases: List of hyperparameter configurations for each training
            stage, in order.

    Returns:
        The trained model.

    """
    for phase in learning_phases:
        logger.info("Starting new training phase.")
        logger.debug("Using phase parameters: {}", phase)

        optimizer = keras.optimizers.SGD(
            learning_rate=phase["learning_rate"], momentum=phase["momentum"]
        )
        model.compile(
            optimizer=optimizer,
            loss={"density_map": SparseMse(), "count": CountAccuracy()},
            loss_weights={
                "density_map": phase["density_map_loss_weight"],
                "count": phase["count_loss_weight"],
            },
        )
        model.fit(
            training_data,
            validation_data=testing_data,
            epochs=phase["num_epochs"],
        )

    return model
