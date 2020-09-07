"""
Nodes for EDA.
"""


import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plot

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
    axes.set(xlabel="Number of Annotations", ylabel="Number of Frames")

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
