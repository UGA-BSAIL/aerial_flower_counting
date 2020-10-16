"""
Defines nodes for the `model_evaluation` pipeline.
"""


import numpy as np
import tensorflow as tf
from PIL import Image
from tabulate import tabulate
from tensorflow import keras

from ...model.inference import count_with_patches
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


def make_example_density_map(
    *,
    model: keras.Model,
    eval_data: tf.data.Dataset,
    patch_scale: float,
    patch_stride: float,
    batch_size: int,
) -> Image.Image:
    """
    Creates an example pseudo-density-map from a trained model.

    Args:
        model: The model to use.
        eval_data: The dataset to get the image from. It will use the first
            item in the dataset.
        patch_scale: The scale factor to apply for the patches we extract.
        patch_stride: The stride to use for extracting patches, provided in
            frame fractions like the scale.
        batch_size: The size of the batches to use for inference.

    Returns:
        The density map, as an image.

    """
    # Obtain the first image.
    first_input_batch, _ = next(iter(eval_data))
    first_image = first_input_batch["image"][0]
    first_image = tf.expand_dims(first_image, axis=0)

    # Calculate the density map.
    density_map = count_with_patches(
        model,
        first_image,
        patch_scale=patch_scale,
        patch_stride=patch_stride,
        batch_size=batch_size,
    )

    # Create a heatmap.
    heatmap = visualize_heat_maps(
        images=first_image,
        features=tf.constant(density_map, dtype=tf.float32),
        max_color_threshold=1.0,
    )

    # Convert to a PIL image.
    heatmap = heatmap[0].numpy()
    return Image.fromarray(heatmap)
