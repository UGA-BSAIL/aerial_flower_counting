"""
Defines nodes for the `model_training` pipeline.
"""

from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Union

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.optimizers.schedules as schedules
from loguru import logger

from ...model.callbacks import LogClassActivations, LogDensityMaps
from ...model.counting_model import build_model
from ...model.losses import make_losses
from ...model.metrics import make_metrics
from ...model.schedules import LoggingWrapper
from ...type_helpers import Vector2I
from ..model_data_load.nodes import DatasetManager


def _make_learning_rate(
    config: Dict[str, Any], *, log_dir: Path
) -> Union[float, schedules.LearningRateSchedule]:
    """
    Creates the learning rate to use for optimization, based on the user
    configuration.

    Args:
        config: The configuration for the learning rate.
        log_dir: The location that we want to write
            tensorboard logs to.

    Returns:
        Either a float for a fixed learning rate, or a `LearningRateSchedule`.

    """
    initial_rate = config["initial"]
    if not config.get("decay", False):
        # No decay is configured.
        logger.debug("Using fixed learning rate of {}.", initial_rate)
        return initial_rate

    logger.debug("Using decaying learning rate.")
    schedule = keras.experimental.CosineDecayRestarts(
        initial_rate,
        config["decay_steps"],
        t_mul=config["t_mul"],
        m_mul=config["m_mul"],
        alpha=config["min_learning_rate"],
    )

    # Create a new directory for Tensorboard logs.
    schedule_log_dir = log_dir / "learning_rate"
    return LoggingWrapper(schedule, log_dir=schedule_log_dir)


def create_model(
    *,
    input_image_shape: Vector2I,
    patch_scale: float,
    sub_patch_scale: float,
    sub_patch_stride: float,
    classify_counts: bool,
    initial_output_bias: float,
) -> keras.Model:
    """
    Builds the model to use.

    Args:
        input_image_shape: The shape of the input images, in the form
        (height, width)
        patch_scale: The scale factor to apply for the patches we extract.
        sub_patch_scale: The scale factor to use for extracting sub-patches when
            computing the cross-scale consistency loss.
        sub_patch_stride: The stride to use for extracting sub-patches when
            computing the cross-scale consistency loss.
        classify_counts: If true, will attempt to classify counts instead of
            regressing them.
        initial_output_bias: The initial bias value to use for the model output.

    Returns:
        The model that it created.

    """
    # Build the model.
    input_height, input_width = input_image_shape
    patch_width = int(input_width * patch_scale)
    patch_height = int(input_height * patch_scale)

    model = build_model(
        input_size=(patch_width, patch_height),
        use_mil=classify_counts,
        sub_patch_scale=sub_patch_scale,
        sub_patch_stride=sub_patch_stride,
        output_bias=initial_output_bias,
    )

    logger.info("Model has {} parameters.", model.count_params())

    return model


def make_callbacks(
    model: keras.Model,
    *,
    testing_data: tf.data.Dataset,
    tensorboard_output_dir: str,
    histogram_frequency: int,
    visualization_period: int,
    max_density_threshold: float,
    classify_counts: bool,
    num_per_batch_to_visualize: int,
    max_num_batches_to_visualize: int,
) -> List[keras.callbacks.Callback]:
    """
    Creates callbacks to use when training the model.

    Args:
        model: The model to train.
        testing_data: The `Dataset` containing pre-processed testing data.
        tensorboard_output_dir: The directory to use for storing Tensorboard
            logs.
        histogram_frequency: Frequency at which to generate histograms for
            Tensorboard output.
        visualization_period: Period in epochs at which to generate density
            map visualizations.
        max_density_threshold: Density threshold to use for colorization.
                Any pixel with this density or more will show up as the maximum
                density color.
        classify_counts: If true, will attempt to classify counts instead of
            regressing them.
        num_per_batch_to_visualize: Number of images to visualize from each
            batch.
        max_num_batches_to_visualize: Maximum number of batches to visualize
            data from.

    Returns:
        The list of callbacks.

    """
    # Create a callback for storing Tensorboard logs.
    log_dir = Path(tensorboard_output_dir) / datetime.now().isoformat()
    logger.debug("Writing Tensorboard logs to {}.", log_dir)
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=histogram_frequency
    )
    callbacks = [tensorboard_callback]

    if not classify_counts:
        # Create a callback for saving density map visualizations.
        density_map_callback = LogDensityMaps(
            log_dir=log_dir / "density_map",
            model=model,
            dataset=testing_data,
            log_period=visualization_period,
            max_density_threshold=max_density_threshold,
            num_images_per_batch=num_per_batch_to_visualize,
        )
        callbacks.append(density_map_callback)
    else:
        # Create a callback for visualizing classification activations.
        activation_callback = LogClassActivations(
            log_dir=log_dir / "class_activations",
            model=model,
            dataset=testing_data,
            log_period=visualization_period,
            num_classes=1,
            num_images_per_batch=num_per_batch_to_visualize,
            max_num_batches=max_num_batches_to_visualize,
        )
        callbacks.append(activation_callback)

    return callbacks


def train_model(
    model: keras.Model,
    *,
    training_data_manager: DatasetManager,
    testing_data: tf.data.Dataset,
    learning_phases: List[Dict[str, Any]],
    classify_counts: bool,
    callbacks: List[keras.callbacks.Callback] = [],
    validation_frequency: int,
    rebalance_frequency: int,
    tensorboard_output_dir: str,
    batch_size: int,
    tag_fraction: float,
) -> keras.Model:
    """
    Trains a model.

    Args:
        model: The model to train.
        training_data_manager: Manager for producing training data.
        testing_data: The `Dataset` containing pre-processed testing data.
        learning_phases: List of hyperparameter configurations for each training
            stage, in order.
        classify_counts: If true, will attempt to classify counts instead of
            regressing them.
        callbacks: The callbacks to use when training.
        validation_frequency: Number of training epochs after which to run
            validation.
        rebalance_frequency: Number of training epochs after which to
            rebalance the dataset.
        tensorboard_output_dir: The directory to use for storing Tensorboard
            logs.
        batch_size: The batch size to use for training.
        tag_fraction: The fraction of the training data to draw from the
            tagged dataset.

    Returns:
        The trained model.

    """
    # Obtain the training data.
    training_data_getter = partial(
        training_data_manager.get_combined,
        batch_size=batch_size,
        tag_fraction=tag_fraction,
    )
    training_data = training_data_getter()

    for phase in learning_phases:
        logger.info("Starting new training phase.")
        logger.debug("Using phase parameters: {}", phase)

        discrete_count_loss_weight = phase.get(
            "discrete_count_loss_weight", 1.0
        )
        # If not specified explicitly, the cross-scale loss is ignored.
        cross_scale_loss_weight = phase.get("cross_scale_loss_weight", 0.0)

        optimizer = keras.optimizers.SGD(
            learning_rate=_make_learning_rate(
                phase["learning_rate"], log_dir=Path(tensorboard_output_dir)
            ),
            momentum=phase["momentum"],
            nesterov=True,
        )
        model.compile(
            optimizer=optimizer,
            loss=make_losses(classify_counts=classify_counts),
            loss_weights={
                "discrete_count": discrete_count_loss_weight,
                "discrete_sub_patch_count": cross_scale_loss_weight,
            },
            metrics=make_metrics(classify_counts=classify_counts),
        )

        ran_epochs = 0
        epochs_to_run = min(phase["num_epochs"], rebalance_frequency)
        while ran_epochs < phase["num_epochs"]:
            model.fit(
                training_data,
                validation_data=testing_data,
                epochs=epochs_to_run,
                callbacks=callbacks,
                validation_freq=validation_frequency,
            )
            ran_epochs += epochs_to_run

            if ran_epochs >= rebalance_frequency:
                logger.info("Re-balancing training data...")
                training_data_manager.rebalance(model=model)
                training_data = training_data_getter()

    return model
