"""
Defines nodes for the `model_training` pipeline.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.optimizers.schedules as schedules
from loguru import logger
from tabulate import tabulate

from .model.callbacks import LogClassActivations, LogDensityMaps
from .model.load_from_dataset import extract_model_input
from .model.losses import CountAccuracy
from .model.sa_net import build_model


def _make_learning_rate(
    config: Dict[str, Any]
) -> Union[float, schedules.LearningRateSchedule]:
    """
    Creates the learning rate to use for optimization, based on the user
    configuration.

    Args:
        config: The configuration for the learning rate.

    Returns:
        Either a float for a fixed learning rate, or a `LearningRateSchedule`.

    """
    initial_rate = config["initial"]
    if not config.get("decay", False):
        # No decay is configured.
        logger.debug("Using fixed learning rate of {}.", initial_rate)
        return initial_rate

    logger.debug("Using decaying learning rate.")
    return keras.experimental.CosineDecayRestarts(
        initial_rate,
        config["decay_steps"],
        t_mul=config["t_mul"],
        m_mul=config["m_mul"],
        alpha=config["min_learning_rate"],
    )


def pre_process_dataset(
    raw_dataset: tf.data.Dataset,
    *,
    patch_scale: float,
    map_height: int,
    map_width: int,
    sigma: int,
    batch_size: int,
    num_prefetch_batches: int,
    allow_randomized: bool,
    bucket_min_values: List[float],
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
        allow_randomized: Whether to allow randomized transformations on this
            dataset that make the output non-deterministic. This includes the
            extraction of random patches as well as the random shuffling of
            data.
        bucket_min_values: A list of the minimum count values that will go in
            each discrete count bucket. Note that the highest bucket will
            contain anything that falls between the largest minimum value and
            infinity.

    Returns:
       A new `Dataset` containing pre-processed data that is ready to use as
       model input.

    """
    extraction_kwargs = dict(
        map_shape=(map_height, map_width),
        sigma=sigma,
        bucket_min_values=bucket_min_values,
        batch_size=batch_size,
        num_prefetch_batches=num_prefetch_batches,
        patch_scale=patch_scale,
        random_patches=allow_randomized,
        shuffle=allow_randomized,
    )
    return extract_model_input(raw_dataset, **extraction_kwargs)


def create_model(
    input_width: int,
    input_height: int,
    patch_scale: float,
    classify_counts: bool,
    bucket_min_values: List[float],
) -> keras.Model:
    """
    Builds the model to use.

    Args:
        input_width: The expected width of the input images, in px.
        input_height: The expected height of the input images, in px.
        patch_scale: The scale factor to apply for the patches we extract.
        classify_counts: If true, will attempt to classify counts instead of
            regressing them.
        bucket_min_values: A list of the minimum count values that will go in
            each discrete count bucket. Note that the highest bucket will
            contain anything that falls between the largest minimum value and
            infinity.

    Returns:
        The model that it created.

    """
    # Build the model.
    patch_width = int(input_width * patch_scale)
    patch_height = int(input_height * patch_scale)

    num_classes = None
    if classify_counts:
        logger.info("Using classification model.")
        num_classes = len(bucket_min_values)
    model = build_model(
        input_size=(patch_width, patch_height), num_classes=num_classes
    )

    logger.info("Model has {} parameters.", model.count_params())

    return model


def _make_losses(
    classify_counts: bool = False,
) -> Dict[str, Union[str, keras.losses.Loss]]:
    """
    Creates the loss dictionary to use when compiling a model.

    Args:
        classify_counts: Whether we are using the classification count output.

    Returns:
        The loss dictionary that it created.

    """
    losses = {"density_map": "mse", "count": CountAccuracy()}
    if classify_counts:
        # Use cross-entropy for classification.
        losses["discrete_count"] = "sparse_categorical_crossentropy"

    return losses


def _make_metrics(
    classify_counts: bool = False,
) -> Dict[str, Union[str, keras.metrics.Metric]]:
    """
    Creates the metrics dictionary to use when compiling a model.

    Args:
        classify_counts: Whether we are using the classification count output.

    Returns:
        The metrics dictionary that it created.

    """
    metrics = {"count": "mean_absolute_error"}

    if classify_counts:
        # Add a standard accuracy metric for the classification.
        metrics["discrete_count"] = "sparse_categorical_accuracy"

    return metrics


def make_callbacks(
    model: keras.Model,
    *,
    testing_data: tf.data.Dataset,
    tensorboard_output_dir: str,
    histogram_frequency: int,
    visualization_period: int,
    max_density_threshold: float,
    bucket_min_values: List[float],
    classify_counts: bool,
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
        bucket_min_values: A list of the minimum count values that will go in
            each discrete count bucket. Note that the highest bucket will
            contain anything that falls between the largest minimum value and
            infinity.
        classify_counts: If true, will attempt to classify counts instead of
            regressing them.

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
        )
        callbacks.append(density_map_callback)
    else:
        # Create a callback for visualizing classification activations.
        activation_callback = LogClassActivations(
            log_dir=log_dir / "class_activations",
            model=model,
            dataset=testing_data,
            log_period=visualization_period,
            num_classes=len(bucket_min_values),
        )
        callbacks.append(activation_callback)

    return callbacks


def train_model(
    model: keras.Model,
    *,
    training_data: tf.data.Dataset,
    testing_data: tf.data.Dataset,
    learning_phases: List[Dict[str, Any]],
    classify_counts: bool,
    callbacks: List[keras.callbacks.Callback] = [],
) -> keras.Model:
    """
    Trains a model.

    Args:
        model: The model to train.
        training_data: The `Dataset` containing pre-processed training data.
        testing_data: The `Dataset` containing pre-processed testing data.
        learning_phases: List of hyperparameter configurations for each training
            stage, in order.
        classify_counts: If true, will attempt to classify counts instead of
            regressing them.
        callbacks: The callbacks to use when training.

    Returns:
        The trained model.

    """
    for phase in learning_phases:
        logger.info("Starting new training phase.")
        logger.debug("Using phase parameters: {}", phase)

        # If not specified explicitly, the discrete count is ignored.
        discrete_count_loss_weight = phase.get(
            "discrete_count_loss_weight", 0.0
        )

        optimizer = keras.optimizers.SGD(
            learning_rate=_make_learning_rate(phase["learning_rate"]),
            momentum=phase["momentum"],
            nesterov=True,
        )
        model.compile(
            optimizer=optimizer,
            loss=_make_losses(classify_counts=classify_counts),
            loss_weights={
                "density_map": phase["density_map_loss_weight"],
                "count": phase["count_loss_weight"],
                "discrete_count": discrete_count_loss_weight,
            },
            metrics=_make_metrics(classify_counts=classify_counts),
        )
        model.fit(
            training_data,
            validation_data=testing_data,
            epochs=phase["num_epochs"],
            callbacks=callbacks,
        )

    return model


def evaluate_model(
    model: keras.Model, *, eval_data: tf.data.Dataset, classify_counts: bool
) -> str:
    """
    Evaluates a model and generates a text report.

    Args:
        model: The model to evaluate.
        eval_data: The data to evaluate the model on.
        classify_counts: If true, will attempt to classify counts instead of
            regressing them.

    Returns:
        A human-readable report of the evaluation results.

    """
    model.compile(
        loss=_make_losses(classify_counts=classify_counts),
        metrics=_make_metrics(classify_counts=classify_counts),
    )

    # Evaluate the model.
    results = model.evaluate(eval_data)

    # Create the report.
    table_rows = []
    for metric_name, metric_value in zip(model.metrics_names, results):
        table_rows.append((metric_name, metric_value))

    return f"Evaluation Results:\n{tabulate(table_rows)}"
