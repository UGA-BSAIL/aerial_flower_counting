import itertools
from typing import Any, Dict, List, Tuple

import numpy as np
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plot

sns.set()


def annotation_histogram(
    cleaned_annotations: List[Tuple[int, List]]
) -> plot.Figure:
    """
    Draws a histogram of the number of annotations per frame.

    Args:
        cleaned_annotations: The cleaned annotation data.

    Returns:
        The `matplotlib` figure containing the histogram.

    """
    logger.info("Generating annotation histogram.")

    # Calculate the number of annotations per frame.
    num_annotations = [len(a) for _, a in cleaned_annotations]

    # Plot it.
    axes = sns.distplot(num_annotations)
    axes.set_title("Number of Annotations per Frame")
    axes.set(xlabel="Number of Annotations", ylabel="Number of Frames")

    return plot.gcf()


def annotation_spatial_dist(
    cleaned_annotations: List[Tuple[int, List]]
) -> plot.Figure:
    """
    Shows the 2D distribution of annotation locations.

    Args:
        cleaned_annotations: The cleaned annotation data.

    Returns:
        The `matplotlib` figure containing the plot.

    """
    logger.info("Generation annotation distribution plot.")

    all_points = []
    for _, annotations in cleaned_annotations:
        all_points.extend([a.points for a in annotations])

    # Flatten the points.
    all_points = list(itertools.chain.from_iterable(all_points))
    all_points = np.array(all_points)

    # Split out the x and y coordinates of the points.
    x_values = all_points[::2]
    y_values = all_points[1::2]

    # Make the plot.
    axes = sns.jointplot(x=x_values, y=y_values, kind="hex")
    axes.fig.suptitle("Distribution of Annotations within the Frame")
    axes.set_axis_labels(xlabel="x (px)", ylabel="y (px)")

    return plot.gcf()
