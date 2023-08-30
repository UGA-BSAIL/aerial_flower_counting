"""
Common functionality between the two versions of the pipeline.
"""
import enum
from datetime import date
from typing import Dict, Iterable, List

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plot


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

    USED_ALTERNATE_ROW = "alternate_row"
    """
    In field layouts that use two rows for each plot, this is a boolean
    variable that specifies whether the "non-default" row was counted.
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


def _add_dap(
    flower_data: pd.DataFrame,
    *,
    field_planted_date: date,
    session_column: enum.Enum,
) -> pd.DataFrame:
    """
    Adds a "days after planting" column to a dataframe with a session column.

    Args:
        flower_data: The counting results dataframe.
        field_planted_date: The date on which the field was planted.
        session_column: The column containing the session data.

    Returns:
        The input, with an added DAP column.

    """
    # Compute DAP.
    seconds_after_planting = flower_data[session_column.value].apply(
        lambda x: (date.fromisoformat(x) - field_planted_date).total_seconds()
    )
    days_after_planting = seconds_after_planting / (60 * 60 * 24)
    days_after_planting = days_after_planting.astype("uint64")
    flower_data[CountingColumns.DAP.value] = days_after_planting

    return flower_data


def add_dap_counting(
    counting_results: pd.DataFrame, *, field_planted_date: date
) -> pd.DataFrame:
    """
    Adds a "days after planting" column to the counting results.

    Args:
        counting_results: The counting results dataframe.
        field_planted_date: The date on which the field was planted.

    Returns:
        The counting results, with an added DAP column.

    """
    return _add_dap(
        counting_results,
        field_planted_date=field_planted_date,
        session_column=CountingColumns.SESSION,
    )


def add_dap_ground_truth(
    ground_truth: pd.DataFrame, *, field_planted_date: date
) -> pd.DataFrame:
    """
    Adds a "days after planting" column to the ground-truth data.

    Args:
        ground_truth: The cleaned ground-truth data.
        field_planted_date: The date on which the field was planted.

    Returns:
        The ground-truth data, with an added DAP column.

    """
    return _add_dap(
        ground_truth,
        field_planted_date=field_planted_date,
        session_column=GroundTruthColumns.SESSION,
    )


def merge_ground_truth(
    *, counting_results: pd.DataFrame, ground_truth: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges the ground-truth and counting results into a single `DataFrame`
    for easy comparison.

    Args:
        counting_results: The complete counting results.
        ground_truth: The cleaned ground-truth data.

    Returns:
        The merged data, indexed by both plot and DAP.

    """
    # Make sure the indices are named the same.
    ground_truth.index.name = counting_results.index.name
    # Merge the data for easy plotting.
    counting_results.set_index(
        CountingColumns.DAP.value, append=True, inplace=True
    )
    ground_truth.set_index(
        GroundTruthColumns.DAP.value, append=True, inplace=True
    )

    merged = pd.merge(
        counting_results,
        ground_truth,
        left_index=True,
        right_index=True,
        how="right",
    )
    # NaN count values are actually zeros. They just don't show up in the
    # original counts dataframe.
    merged[CountingColumns.COUNT.value] = merged[
        CountingColumns.COUNT.value
    ].fillna(0)
    return merged


def plot_ground_truth_regression(counts_with_gt: pd.DataFrame) -> plot.Figure:
    """
    Plots the regression between the counting results and the ground-truth
    counts.

    Args:
        counts_with_gt: The merged counting results and ground-truth data.

    Returns:
        The plot that it created.

    """
    # Plot the regression.
    axes = sns.residplot(
        data=counts_with_gt,
        x=GroundTruthColumns.TRUE_COUNT.value,
        y=CountingColumns.COUNT.value,
    )
    axes.set_title("Counting Residuals")
    axes.set(xlabel="Ground-Truth", ylabel="Automatic")

    return plot.gcf()


def merge_dicts(*args: Dict) -> Dict:
    """
    Small helper node to merge dictionaries.

    Args:
        *args: The dictionaries to merge.

    Returns:
        The merged dictionary.

    """
    merged = {}
    for arg in args:
        merged.update(arg)
    return merged
