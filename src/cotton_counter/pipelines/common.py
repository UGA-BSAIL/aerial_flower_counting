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


def collect_session_results(*session_results: pd.DataFrame,) -> pd.DataFrame:
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
