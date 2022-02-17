"""
Contains nodes for the `count_plots` pipeline.
"""


import enum
from datetime import date
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plot
from pydantic.dataclasses import dataclass
from yolov5 import YOLOv5

_PREDICTION_BATCH_SIZE = 50
"""
Size of batches to use for prediction. This mostly impacts memory use.
"""


sns.set()


@enum.unique
class GenotypeColumns(enum.Enum):
    """
    Names of the columns in the genotype table.
    """

    SERIAL_NUM = "SN"
    """
    Serial number of the plot.
    """
    GENOTYPE = "Genotype"
    """
    Genotype identifier.
    """
    PLOT_NUM = "2021 IDENTIFIER #"
    """
    Plot that this genotype is planted in.
    """
    POPULATION = "Population"
    """
    Genotype population.
    """


@enum.unique
class DetectionColumns(enum.Enum):
    """
    Names of the columns in the detection results dataframe.
    """

    SESSION = "session"
    """
    The name of the session that these detections are from.
    """
    DETECTION_PLOT = "plot"
    """
    The (detection) plot number that these detections are from.
    """
    CONFIDENCE = "confidence"
    """
    The confidence score of the detection.
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
    PLOT = "plot"
    """
    The field plot number that this count is from.
    """
    COUNT = "count"
    """
    The number of flowers in this plot.
    """
    DAP = "days_after_planting"
    """
    The number of days after planting that this count was taken at.
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
            image_results[DetectionColumns.DETECTION_PLOT.value] = plot_num
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
    return counting_results[
        counting_results[DetectionColumns.CONFIDENCE.value] >= min_confidence
    ]


def compute_counts(
    detection_results: pd.DataFrame, *, field_config: FieldConfig
) -> pd.DataFrame:
    """
    Computes the counts for each plot based on the detection results.

    Args:
        detection_results: The results from the detection stage.
        field_config: Represents the configuration of the field.

    Returns:
        A DataFrame of the computed counts.

    """
    # Figure out how many detections we have for each plot.
    counts_per_plot_series = detection_results.value_counts(
        subset=[
            DetectionColumns.SESSION.value,
            DetectionColumns.DETECTION_PLOT.value,
        ]
    )
    # Convert the series to a frame.
    counts_per_plot = counts_per_plot_series.index.to_frame(index=False)
    counts_per_plot[
        CountingColumns.COUNT.value
    ] = counts_per_plot_series.values

    # Convert the plot numbers used during detection to the plot numbers used
    # in the field.
    counts_per_plot[CountingColumns.PLOT.value] = counts_per_plot[
        DetectionColumns.DETECTION_PLOT.value
    ].apply(_to_field_plot_num, field_config=field_config)

    # Remove null values.
    counts_per_plot.dropna(inplace=True)
    # Use plot number as an index.
    counts_per_plot.set_index(CountingColumns.PLOT.value, inplace=True)
    counts_per_plot.sort_index(inplace=True)

    return counts_per_plot


def add_dap(
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
    # Compute DAP.
    seconds_after_planting = counting_results[
        CountingColumns.SESSION.value
    ].apply(
        lambda x: (date.fromisoformat(x) - field_planted_date).total_seconds()
    )
    days_after_planting = seconds_after_planting / (60 * 60 * 24)
    days_after_planting = days_after_planting.astype("uint64")
    counting_results[CountingColumns.DAP.value] = days_after_planting

    return counting_results


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
    # Convert to a dataframe where we have one column per session.
    counts_by_session = pd.DataFrame(index=counting_results.index.unique())
    sessions = counting_results[CountingColumns.SESSION.value].unique()
    for session in sessions:
        # Get the plot counts.
        session_counts = counting_results[
            counting_results[CountingColumns.SESSION.value] == session
        ]
        counts_by_session[session] = session_counts[
            CountingColumns.COUNT.value
        ]

    # Plots with no flowers will be set to NA. Fill these in with zero.
    counts_by_session.fillna(0, inplace=True)
    # Sort the columns by date.
    counts_by_session = counts_by_session.reindex(
        sorted(counts_by_session.columns), axis=1
    )
    # Turn the plot index into a column so that it shows up in the spreadsheet.
    counts_by_session.insert(0, "Plot", counts_by_session.index.values)

    return counts_by_session


def clean_genotypes(raw_genotypes: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans up the genotype data that was loaded from the genotype spreadsheet.

    Args:
        raw_genotypes: The raw genotype information that was loaded.

    Returns:
        The cleaned genotype table.

    """
    # Remove the serial number column.
    cleaned = raw_genotypes.drop(columns=[GenotypeColumns.SERIAL_NUM.value])

    # Index by plot number.
    cleaned.set_index(GenotypeColumns.PLOT_NUM.value, inplace=True)
    cleaned.sort_index(inplace=True)

    return cleaned


def compute_flowering_peak(counting_results: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the peak flowering times for each plot.

    Args:
        counting_results: The complete counting results.

    Returns:
        A dataframe in the same format as the input, but containing only the
        rows that represent the peak flower count for each plot.

    """
    # Find the rows with the maximum count for each plot.
    plot_groups = counting_results.groupby([counting_results.index])
    return plot_groups.apply(
        lambda g: g.iloc[g[CountingColumns.COUNT.value].argmax()]
    )


def plot_peak_flowering_dist(
    *, peak_flowering_times: pd.DataFrame, genotypes: pd.DataFrame
) -> plot.Figure:
    """
    Draws a histogram of the peak flowering time for each plot. The histogram
    will be colorized by population.

    Args:
        peak_flowering_times: The dataframe containing computed peak
            flowering times.
        genotypes: The dataframe containing genotype information.

    Returns:
        The plot that it made.

    """
    # Merge flowering and genotype data together for easy plotting.
    combined_data = pd.merge(
        peak_flowering_times, genotypes, left_index=True, right_index=True
    )

    # Plot it.
    axes = sns.histplot(
        data=combined_data,
        x=CountingColumns.DAP.value,
        hue=GenotypeColumns.POPULATION.value,
        multiple="stack",
    )
    axes.set_title("Peak Flowering Times")
    axes.set(xlabel="Days After Planting", ylabel="Count")

    return plot.gcf()
