"""
Contains nodes for the `count_plots` pipeline.
"""


from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass
from yolov5 import YOLOv5

_PREDICTION_BATCH_SIZE = 50
"""
Size of batches to use for prediction. This mostly impacts memory use.
"""


@dataclass
class FieldConfig:
    """
    Represents the configuration of the actual field.

    Attributes:
        num_rows: The total number of rows.

        first_row_num: The number assigned to the first row.
        first_plot_num: The number assigned to the first plot in each row.

        empty_rows: Set of rows that are not planted.
    """

    num_rows: int

    first_row_num: int
    first_plot_num: int

    empty_rows: Set[int] = frozenset()


def _batch_iter(iterable: Iterable, *, batch_size: int) -> Iterable[List]:
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


def _to_field_plot_num(
    plot_num: int, *, field_config: FieldConfig
) -> Optional[int]:
    """
    Converts the plot numbers used by the detection to the plot number
    format used on the plot labels in the field.

    Args:
        plot_num: The detection plot number.
        field_config: The layout of the field.

    Returns:
        The field plot number, or None if there is no corresponding-numbered
        plot.

    """
    # The plot extraction method numbers plots in row-major order starting
    # from zero in the upper left.
    row_num = plot_num // field_config.num_rows
    column_num = plot_num % field_config.num_rows

    # Assign the correct row and plot numbers.
    row_num += field_config.first_row_num
    column_num += field_config.first_plot_num

    if row_num in field_config.empty_rows:
        # This row is not planted.
        return None

    # Create the complete plot number.
    return int(f"{row_num}{column_num}")


def detect_flowers(
    images: Iterable[np.ndarray],
    *,
    weights_file: Path,
    session_name: str,
    image_size: Tuple[int, int] = (640, 640),
) -> pd.DataFrame:
    """
    Detects the flowers in a series of input images.

    Args:
        images: The images to detect flowers in.
        weights_file: The location of the file to load the model weights from.
        session_name: The name of this session.
        image_size: The model input size to use for inference.

    Returns:
        The detections for all the images, as a Pandas dataframe.

    """
    # Load the model.
    model = YOLOv5(weights_file)

    # Infer on batches.
    results = []
    plot_num = 0
    for batch in _batch_iter(images, batch_size=_PREDICTION_BATCH_SIZE):
        batch_results = (
            model.predict(batch, size=max(image_size)).pandas().xyxy
        )

        # Add a new column indicating the source image.
        for image_results in batch_results:
            image_results["plot"] = plot_num
            plot_num += 1

        results.extend(batch_results)

    all_results = pd.concat(results, ignore_index=True)
    # Add a column for the session name.
    all_results["session"] = session_name
    return all_results


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
    counting_results: pd.DataFrame, *, min_confidence: float
) -> pd.DataFrame:
    """
    Filters the counting results to remove any detections with low confidence.

    Args:
        counting_results: The complete counting results.
        min_confidence: The minimum confidence value to keep.

    Returns:
        The filtered results.

    """
    return counting_results[counting_results["confidence"] >= min_confidence]


def create_per_plot_table(
    counting_results: pd.DataFrame, *, field_config: FieldConfig
) -> pd.DataFrame:
    """
    Converts the counting results to a table with per-plot counts.

    Args:
        counting_results: The complete counting results.
        field_config: Represents the configuration of the field.

    Returns:
        A table, where each row is a field plot number, and each column
        represents a particular session. The corresponding cell contains the
        count for that plot during that session.

    """
    # Figure out how many detections we have for each plot.
    counts_per_plot_series = counting_results.value_counts(
        subset=["session", "plot"]
    )
    # Convert the series to a frame.
    counts_per_plot = counts_per_plot_series.index.to_frame(index=False)
    counts_per_plot["count"] = counts_per_plot_series.values

    # Convert the plot numbers used during detection to the plot numbers used
    # in the field.
    counts_per_plot["plot"] = counts_per_plot["plot"].apply(
        _to_field_plot_num, field_config=field_config
    )
    # Remove null values.
    counts_per_plot.dropna(inplace=True)
    # Use plot number as an index.
    counts_per_plot.set_index("plot", inplace=True)
    counts_per_plot.sort_index(inplace=True)

    # Convert to a dataframe where we have one column per session.
    counts_by_session = pd.DataFrame(index=counts_per_plot.index.unique())
    sessions = counts_per_plot["session"].unique()
    for session in sessions:
        # Get the plot counts.
        session_counts = counts_per_plot[counts_per_plot["session"] == session]
        counts_by_session[session] = session_counts["count"]

    # Plots with no flowers will be set to NA. Fill these in with zero.
    counts_by_session.fillna(0, inplace=True)
    # Sort the columns by date.
    counts_by_session = counts_by_session.reindex(
        sorted(counts_by_session.columns), axis=1
    )
    # Turn the plot index into a column so that it shows up in the spreadsheet.
    counts_by_session.insert(0, "Plot", counts_by_session.index.values)

    return counts_by_session
