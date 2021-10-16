"""
Nodes for EDA.
"""


import io
from typing import Iterable

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from loguru import logger
from matplotlib import pyplot as plot
from PIL import Image, ImageDraw

sns.set()


def annotation_histogram(local_annotations: pd.DataFrame) -> plot.Figure:
    """
    Draws a histogram of the number of annotations per frame.

    Args:
        local_annotations: The cleaned annotation data, as a `DataFrame`.

    Returns:
        The `matplotlib` figure containing the histogram.

    """
    logger.info("Generating annotation histogram.")

    # Calculate the number of annotations per frame.
    annotations_per_frame = local_annotations.pivot_table(
        index=["frame_num"], aggfunc="size"
    )

    # Plot it.
    axes = sns.distplot(annotations_per_frame)
    axes.set_title("Number of Annotations per Frame")
    axes.set(xlabel="Number of Annotations", ylabel="Normalized Density")

    return plot.gcf()


def annotation_spatial_dist(local_annotations: pd.DataFrame) -> plot.Figure:
    """
    Shows the 2D distribution of annotation locations.

    Args:
        local_annotations: The cleaned annotation data, as a `DataFrame`.

    Returns:
        The `matplotlib` figure containing the plot.

    """
    logger.info("Generation annotation distribution plot.")

    # Make the plot.
    axes = sns.jointplot(
        x=local_annotations["x"], y=local_annotations["y"], kind="hex"
    )
    axes.fig.suptitle("Distribution of Annotations within the Frame")
    axes.set_axis_labels(xlabel="x (px)", ylabel="y (px)")

    return plot.gcf()


_DOT_RADIUS = 40
"""
Radius of the annotation dots to plot, in pixels.
"""


def _plot_annotations(
    image: Image.Image, annotation_x: np.ndarray, annotation_y: np.ndarray
) -> None:
    """
    Plots annotations on top of an image.

    Args:
        image: The image to plot annotations on.
        annotation_x: The annotation x-coordinates.
        annotation_y: The annotation y-coordinates.

    """
    draw = ImageDraw.Draw(image)

    # Convert annotations to pixels.
    image_width, image_height = image.size
    annotation_x *= image_width
    annotation_y *= image_height

    for x, y in zip(annotation_x, annotation_y):
        top_left = (x - _DOT_RADIUS, y - _DOT_RADIUS)
        bottom_right = (x + _DOT_RADIUS, y + _DOT_RADIUS)
        draw.ellipse([top_left, bottom_right], fill="red")


def visualize_ground_truth(dataset: tf.data.Dataset) -> Iterable[Image.Image]:
    """
    Visualizes the ground-truth annotations for a set of data.

    Args:
        dataset: The dataset to visualize. Should be in a parsed form, but
            still with point annotations.

    Returns:
        The visualizations it generated, as images.

    """
    for feature_dict in dataset:
        image_batch = feature_dict["image"]
        annotation_x_batch = feature_dict["annotation_x"]
        annotation_y_batch = feature_dict["annotation_y"]

        for image, annotation_x, annotation_y in zip(
            image_batch, annotation_x_batch, annotation_y_batch
        ):
            binary_image = io.BytesIO(image[0].numpy())
            pil_image = Image.open(binary_image)

            _plot_annotations(
                pil_image, annotation_x.numpy(), annotation_y.numpy()
            )

            yield pil_image
