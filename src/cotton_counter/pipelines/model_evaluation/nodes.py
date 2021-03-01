"""
Defines nodes for the `model_evaluation` pipeline.
"""


from typing import Any, Dict, Iterable

import numpy as np
import sklearn.metrics
import tensorflow as tf
from loguru import logger
from matplotlib import pyplot as plot
from PIL import Image
from tabulate import tabulate
from tensorflow import keras

from ...model.inference import calculate_max_density, count_with_patches
from ...model.losses import make_losses
from ...model.metrics import make_metrics
from ...model.visualization import visualize_heat_maps


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
        loss=make_losses(classify_counts=classify_counts),
        metrics=make_metrics(classify_counts=classify_counts),
    )

    # Evaluate the model.
    results = model.evaluate(eval_data)

    # Create the report.
    table_rows = []
    for metric_name, metric_value in zip(model.metrics_names, results):
        table_rows.append((metric_name, metric_value))

    return f"Evaluation Results:\n{tabulate(table_rows)}\n"


def make_example_density_maps(
    *,
    model: keras.Model,
    eval_data: tf.data.Dataset,
    patch_scale: float,
    patch_stride: float,
    batch_size: int,
) -> Iterable[Image.Image]:
    """
    Creates an example pseudo-density-map from a trained model.

    Args:
        model: The model to use.
        eval_data: The dataset to get the images from.
        patch_scale: The scale factor to apply for the patches we extract.
        patch_stride: The stride to use for extracting patches, provided in
            frame fractions like the scale.
        batch_size: The size of the batches to use for inference.

    Yields:
        The density map for each image in the dataset.

    """
    for batch, _ in eval_data:
        # Calculate the density map.
        images = batch["image"]
        density_maps = count_with_patches(
            model,
            images,
            patch_scale=patch_scale,
            patch_stride=patch_stride,
            batch_size=batch_size,
        )

        # Create a heatmap.
        max_density = calculate_max_density(images, patch_scale=patch_scale)
        heatmaps = visualize_heat_maps(
            images=images,
            features=tf.constant(density_maps, dtype=tf.float32),
            max_color_threshold=max_density,
        )

        # Convert to a PIL image.
        heatmaps = heatmaps.numpy()
        for heatmap in heatmaps:
            yield Image.fromarray(heatmap)


def _make_report(*, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Generates a machine-readable report for a regression problem. It contains
    the per-sample error, as well as the mean overall error.

    Args:
        y_true: The true values.
        y_pred: The predictions.

    Returns:
        The generated report.

    """
    per_sample_error = np.abs(y_true - y_pred)

    # Generate per-sample error lines.
    per_sample_lines = []
    for sample_num, (true, pred, error) in enumerate(
        zip(y_true, y_pred, per_sample_error)
    ):
        per_sample_lines.append(
            dict(
                sample_num=sample_num,
                y_true=float(true),
                y_pred=float(pred),
                error=float(error),
            )
        )

    # Calculate overall error.
    mean_error = np.mean(per_sample_error)
    logger.info("Mean absolute count error: {}", mean_error)

    return dict(
        mean_error=mean_error.tolist(), per_sample_error=per_sample_lines
    )


def estimate_counting_accuracy(
    *,
    model: keras.Model,
    eval_data: tf.data.Dataset,
    patch_scale: float,
    patch_stride: float,
    batch_size: int,
) -> Dict[str, Any]:
    """
    Estimates the pure counting accuracy of the model on a dataset.

    Args:
        model: The model to use.
        eval_data: The dataset to get the images from. It should produce full
            images, and not patches. It should also contain a raw count target.
        patch_scale: The scale factor to apply for the patches we extract.
        patch_stride: The stride to use for extracting patches, provided in
            frame fractions like the scale.
        batch_size: The size of the batches to use for inference.

    Returns:
        A dictionary containing a machine-readable accuracy report.

    """
    # Store the predicted and actual counts for each image.
    predicted_counts = []
    actual_counts = []

    eval_data = eval_data.unbatch().batch(batch_size)

    for input_batch, target_batch in eval_data:
        # Calculate the density maps.
        density_maps = count_with_patches(
            model,
            input_batch["image"],
            patch_scale=patch_scale,
            patch_stride=patch_stride,
            batch_size=batch_size,
        )

        # Estimate the predicted counts for each image.
        batch_predicted_counts = np.sum(density_maps, axis=(1, 2, 3))
        predicted_counts.append(batch_predicted_counts)

        # Save the actual counts too.
        actual_counts.append(target_batch["count"])

    # Calculate an overall count error.
    predicted_counts = np.concatenate(predicted_counts, axis=0)
    actual_counts = np.concatenate(actual_counts, axis=0)
    logger.debug("Generated counts for {} images.", len(predicted_counts))

    return _make_report(y_true=actual_counts, y_pred=predicted_counts)


def plot_roc_curve(
    model: keras.Model, *, eval_data: tf.data.Dataset, classify_counts: bool
) -> plot.Figure:
    """
    Plots an ROC curve for the model.

    Args:
        model: The model to generate the ROC curve for.
        eval_data: The dataset we want to use for generating the ROC curve.
        classify_counts: If true, will attempt to classify counts instead of
            regressing them.

    Returns:
        The figure in which it plotted the ROC curve.

    """
    model.compile(
        loss=make_losses(classify_counts=classify_counts),
        metrics=make_metrics(classify_counts=classify_counts),
    )

    # Obtain the labels and predictions.
    labels = [t["discrete_count"].numpy() for _, t in eval_data.unbatch()]
    model_results = model.predict(eval_data)
    probabilities = model_results["discrete_count"].squeeze()

    # Convert so that the positive class is 1.
    labels = 1 - np.array(labels)
    probabilities = 1.0 - probabilities

    # Plot the curve.
    false_positives, true_positives, _ = sklearn.metrics.roc_curve(
        labels, probabilities
    )

    plot.plot(
        100 * false_positives,
        100 * true_positives,
        label="ROC Curve",
        linewidth=2,
    )
    plot.xlabel("False Positives [%]")
    plot.ylabel("True Positives [%]")
    plot.grid(True)
    axes = plot.gca()
    axes.set_aspect("equal")

    return plot.gcf()
