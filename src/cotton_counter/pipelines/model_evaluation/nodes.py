"""
Defines nodes for the `model_evaluation` pipeline.
"""


from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
import tensorflow as tf
from loguru import logger
from matplotlib import pyplot as plot
from PIL import Image
from tabulate import tabulate
from tensorflow import keras

from ...model.inference import (
    calculate_max_density,
    count_with_patches,
    predict_with_activation_maps_patched,
)
from ...model.losses import make_losses
from ...model.metrics import make_metrics
from ...model.visualization import visualize_heat_maps

sns.set()


def evaluate_model(model: keras.Model, *, eval_data: tf.data.Dataset) -> str:
    """
    Evaluates a model and generates a text report.

    Args:
        model: The model to evaluate.
        eval_data: The data to evaluate the model on.

    Returns:
        A human-readable report of the evaluation results.

    """
    model.compile(
        # Disable focal loss for the evaluation.
        loss=make_losses(alpha=1.0, gamma=0.0),
        metrics=make_metrics(include_count=True),
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
    # Use the specified evaluation batch size.
    eval_data = eval_data.unbatch().batch(batch_size)

    for batch, _ in eval_data:
        # Calculate the density map.
        images = batch["image"]
        activation_maps = predict_with_activation_maps_patched(
            model, images, batch_size=batch_size
        )
        density_maps = count_with_patches(
            activation_maps,
            patch_scale=patch_scale,
            patch_stride=patch_stride,
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


def estimate_counting_accuracy(
    *,
    model: keras.Model,
    eval_data: tf.data.Dataset,
    batch_size: int,
    patch_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates the pure counting accuracy of the model on a dataset.

    Args:
        model: The model to use.
        eval_data: The dataset to get the images from. It should contain a raw
            count target.
        batch_size: The size of the batches to use for inference.
        patch_scale: The patch scale that was used when extracting patches in
            `eval_data`.

    Returns:
        The true counts and the corresponding predicted counts.

    """
    # We get activation maps and then combine them before manually taking the
    # average to find the counts. This eliminates any bias from predicting
    # with a smaller patch size.
    activation_layer = model.get_layer("activation_maps_count")
    activation_output = activation_layer.get_output_at(0)
    activation_model = tf.keras.Model(
        inputs=model.inputs, outputs=[activation_output]
    )

    # Get the predicted counts for the data.
    eval_data = eval_data.unbatch().batch(batch_size)
    activation_maps = activation_model.predict(eval_data)

    # Save the actual counts too.
    actual_counts = []
    for _, target_batch in eval_data:
        actual_counts.append(target_batch["count"])
    actual_counts = np.concatenate(actual_counts, axis=0)

    # Group patch results by corresponding input images and compute overall
    # counts.
    num_patches_per_image = int((1.0 / patch_scale) ** 2)
    logger.debug("Num patches/image: {}", num_patches_per_image)
    actual_counts = np.reshape(actual_counts, (num_patches_per_image, -1))
    actual_counts = np.sum(actual_counts, axis=0)
    grouped_maps = np.reshape(
        activation_maps,
        (-1, num_patches_per_image,) + activation_maps.shape[1:],
    )
    predicted_counts = np.mean(grouped_maps, axis=(1, 2, 3, 4))
    logger.debug("Generated counts for {} images.", len(actual_counts))

    return actual_counts, predicted_counts


def make_accuracy_report(
    *, y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Any]:
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


def make_counting_histogram(
    *, y_true: np.ndarray, y_pred: np.ndarray
) -> plot.Figure:
    """
    Plots a histogram of the counting accuracy over the entire dataset.

    Args:
        y_true: The true counts for each image.
        y_pred: The predicted counts for each image.

    Returns:
        The figure in which it plotted this histogram.

    """
    # Calculate the counting errors.
    errors = y_true - y_pred

    # Plot the histogram.
    axes = sns.histplot(errors, stat="count")
    axes.set(xlabel="Count Error", ylabel="Number of Images")

    return plot.gcf()


def calculate_roc_points(
    model: keras.Model, *, eval_data: tf.data.Dataset
) -> pd.DataFrame:
    """
    Calculates the points on an ROC.

    Args:
        model: The model to generate the ROC curve for.
        eval_data: The dataset we want to use for generating the ROC curve.

    Returns:
        The figure in which it plotted the ROC curve.

    """
    model.compile(
        # Disable focal loss for evaluation.
        loss=make_losses(alpha=1.0, gamma=0.0),
        metrics=make_metrics(),
    )

    # Obtain the labels and predictions.
    labels = [t["has_flower"].numpy() for _, t in eval_data.unbatch()]
    model_results = model.predict(eval_data)
    probabilities = model_results["has_flower"].squeeze()

    # Calculate the curve points.
    false_positives, true_positives, _ = sklearn.metrics.roc_curve(
        labels, probabilities
    )

    return pd.DataFrame(
        {"false_positives": false_positives, "true_positives": true_positives}
    )


def plot_roc_curve(roc_points: pd.DataFrame) -> plot.Figure:
    """
    Plots an ROC curve for the model.

    Args:
        roc_points: The `DataFrame` containing the points in the ROC curve to
            plot.

    Returns:
        The figure in which it plotted the ROC curve.

    """
    false_positives = roc_points["false_positives"]
    true_positives = roc_points["true_positives"]

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
