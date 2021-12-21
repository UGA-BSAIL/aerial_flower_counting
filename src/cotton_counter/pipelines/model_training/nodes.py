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


def set_check_numerics(enable: bool) -> None:
    """
    Sets whether to enable checking for NaNs and infinities.

    Args:
        enable: If true, will enable the checks.

    """
    if enable:
        logger.info("Enabling numeric checks. Training might be slow.")
        tf.debugging.enable_check_numerics()
    else:
        tf.debugging.disable_check_numerics()


def create_model(
    *,
    input_image_shape: Vector2I,
    patch_scale: float,
    num_loss_scales: int,
    initial_output_bias: float,
    focal_loss_alpha: float,
    focal_loss_gamma: float,
) -> keras.Model:
    """
    Builds the model to use.

    Args:
        input_image_shape: The shape of the input images, in the form
        (height, width)
        patch_scale: The scale factor to apply for the patches we extract.
        num_loss_scales: The number of different scales to use when computing
            the loss.
        initial_output_bias: The initial bias value to use for the model output.
        focal_loss_alpha: The alpha parameter to use for the focal loss
            between the PAC outputs and counts.
        focal_loss_gamma: The gamma parameter to use for the focal loss
            between the PAC outputs and counts.

    Returns:
        The model that it created.

    """
    # Build the model.
    input_height, input_width = input_image_shape
    patch_width = int(input_width * patch_scale)
    patch_height = int(input_height * patch_scale)

    model = build_model(
        input_size=(patch_width, patch_height),
        num_scales=num_loss_scales,
        output_bias=initial_output_bias,
        focal_alpha=focal_loss_alpha,
        focal_gamma=focal_loss_gamma,
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
    callbacks = [tensorboard_callback, tf.keras.callbacks.TerminateOnNaN()]

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
    callbacks: List[keras.callbacks.Callback] = [],
    validation_frequency: int,
    rebalance_frequency: int,
    tensorboard_output_dir: str,
    batch_size: int,
    tag_fraction: float,
    focal_loss_alpha: float,
    focal_loss_gamma: float,
) -> keras.Model:
    """
    Trains a model.

    Args:
        model: The model to train.
        training_data_manager: Manager for producing training data.
        testing_data: The `Dataset` containing pre-processed testing data.
        learning_phases: List of hyperparameter configurations for each training
            stage, in order.
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
        focal_loss_alpha: Alpha parameter for focal loss.
        focal_loss_gamma: Gamma parameter for focal loss.

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

        pac_loss_weight = phase.get("pac_loss_weight", 1.0)
        combined_bce_loss_weight = phase.get("combined_bce_loss_weight", 1.0)
        scale_consistency_loss_weight = phase.get(
            "scale_consistency_loss_weight", 1.0
        )

        optimizer = keras.optimizers.SGD(
            learning_rate=_make_learning_rate(
                phase["learning_rate"], log_dir=Path(tensorboard_output_dir)
            ),
            momentum=phase["momentum"],
            nesterov=True,
        )
        model.compile(
            optimizer=optimizer,
            loss=make_losses(alpha=focal_loss_alpha, gamma=focal_loss_gamma,),
            loss_weights={
                "has_flower": pac_loss_weight,
                "combined_bce_loss": combined_bce_loss_weight,
                "scale_consistency_loss": scale_consistency_loss_weight,
            },
            metrics=make_metrics(),
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
