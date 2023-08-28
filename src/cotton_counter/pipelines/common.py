"""
Common functionality between the two versions of the pipeline.
"""
import enum
from typing import Iterable, List

import pandas as pd


def batch_iter(iterable: Iterable, *, batch_size: int) -> Iterable[List]:
    """
    Condenses an iterable into batches.

    Args:
        iterable: The iterable.
        batch_size: The batch size to use.

    Yields:
        The batches that it creates.

    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    # Produce a partial batch at the end if we have one.
    if len(batch) > 0:
        yield batch


@enum.unique
class DetectionColumns(enum.Enum):
    """
    Names of the columns in the detection results dataframe.
    """

    SESSION = "session"
    """
    The name of the session that these detections are from.
    """
    IMAGE_ID = "image_id"
    """
    Unique identifier for the image that these detections are from.
    """
    PLOT_NUM = "plot_num"
    """
    The plot number that these detections are from.
    """
    BOX_NUM = "box_num"
    """
    The number of the box that these detections are from.
    """
    CONFIDENCE = "confidence"
    """
    The confidence score of the detection.
    """

    X1 = "x1"
    """
    The x-coordinate of the top left corner of the detection.
    """
    Y1 = "y1"
    """
    The y-coordinate of the top left corner of the detection.
    """
    X2 = "x2"
    """
    The x-coordinate of the bottom right corner of the detection.
    """
    Y2 = "y2"
    """
    The y-coordinate of the bottom right corner of the detection.
    """


@enum.unique
class GroundTruthColumns(enum.Enum):
    """
    Names of the columns in the ground-truth table.
    """

    PLOT = "Plot Number"
    """
    The field plot number.
    """
    TRUE_COUNT = "GT Count"
    """
    The ground-truth count for that plot.
    """
    MANUAL_IMAGE_COUNT = "Drone Manual"
    """
    The number of flowers counted manually from the drone imagery.
    """
    SESSION = "Date"
    """
    The date for the session that the data was collected on.
    """
    DAP = "days_after_planting"
    """
    The number of days after planting that this count was taken at.
    """
    PERSON = "Person measured"
    """
    The initials of the person who took the GT measurement.
    """


@enum.unique
class CountingColumns(enum.Enum):
    """
    Names of the columns in the counting results dataframe.
    """

    # This is carried over directly from the detection results.
    SESSION = DetectionColumns.SESSION.value
    """
    The name of the session that this count is from.
    """
    PLOT = "field_plot"
    """
    The field plot number that this count is from.
    """
    COUNT = "count"
    """
    The number of flowers in this plot.
    """
    DAP = GroundTruthColumns.DAP.value
    """
    The number of days after planting that this count was taken at.
    """


def collect_session_results(
    *session_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Collects the results from an entire set of sessions into a single Pandas
    DataFrame.

    Args:
        session_results: The results from each session.

    Returns:
        A DataFrame containing all the results.

    """
    return pd.concat(session_results, ignore_index=True)


def filter_low_confidence(
    detection_results: pd.DataFrame, *, min_confidence: float
) -> pd.DataFrame:
    """
    Filters the counting results to remove any detections with low confidence.

    Args:
        detection_results: The complete detection results.
        min_confidence: The minimum confidence value to keep.

    Returns:
        The filtered results.

    """
    return detection_results[
        detection_results[DetectionColumns.CONFIDENCE.value] >= min_confidence
    ]


def compute_counts(detection_results: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the counts for each plot based on the detection results.

    Args:
        detection_results: The results from the detection stage.

    Returns:
        A DataFrame of the computed counts.

    """
    # Remove any rows with NaN plot numbers, which means that those
    # detections fell outside any marked plot.
    detection_results.dropna(subset=[DetectionColumns.PLOT_NUM.value])

    # Figure out how many detections we have for each plot.
    counts_per_plot_series = detection_results.value_counts(
        subset=[
            DetectionColumns.SESSION.value,
            DetectionColumns.PLOT_NUM.value,
        ]
    )
    # Convert the series to a frame.
    counts_per_plot = counts_per_plot_series.index.to_frame(index=False)
    counts_per_plot[
        CountingColumns.COUNT.value
    ] = counts_per_plot_series.values

    return counts_per_plot
