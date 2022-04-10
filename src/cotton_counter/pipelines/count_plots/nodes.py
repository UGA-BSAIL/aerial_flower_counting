"""
Contains nodes for the `count_plots` pipeline.
"""


import enum
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as sm
from matplotlib import pyplot as plot
from pandarallel import pandarallel
from pydantic.dataclasses import dataclass
from yolov5 import YOLOv5

_PREDICTION_BATCH_SIZE = 50
"""
Size of batches to use for prediction. This mostly impacts memory use.
"""


sns.set()

pandarallel.initialize()


@enum.unique
class GenotypeColumns(enum.Enum):
    """
    Names of the columns in the genotype table.
    """

    GENOTYPE = "Genotype"
    """
    Genotype identifier.
    """
    PLOT = "2021 IDENTIFIER #"
    """
    Plot that this genotype is planted in.
    """
    POPULATION = "Population"
    """
    Genotype population.
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
    DAP = GroundTruthColumns.DAP.value
    """
    The number of days after planting that this count was taken at.
    """


@enum.unique
class HeightColumns(enum.Enum):
    """
    Names of the columns in the plot height dataframe.
    """

    PLOT = CountingColumns.PLOT.value
    """
    The field plot number that these heights are from.
    """
    HEIGHT = "height"
    """
    The computed height value for this plot.
    """


@enum.unique
class FloweringSlopeColumns(enum.Enum):
    """
    Names of the columns in the flowering slope dataframe.
    """

    SLOPE = "slope"
    """
    The slope of the line fitted to the flowering curve.
    """
    INTERCEPT = "intercept"
    """
    The intercept of the line fitted to the flowering curve.
    """


@enum.unique
class FloweringTimeColumns(enum.Enum):
    """
    Names of the columns in the flowering time dataframe.
    """

    DURATION = "duration"
    """
    The total flowering duration, in days.
    """


@dataclass
class FieldConfig:
    """
    Represents the configuration of the actual field.

    Attributes:
        num_rows: The total number of plots in each row.

        first_row_num: The number assigned to the first row.
        first_plot_num: The number assigned to the first plot in each row.

        empty_rows: Set of rows that are not planted.
    """

    num_plots: int

    first_row_num: int
    first_plot_num: int

    empty_rows: Set[int] = frozenset()

    @classmethod
    def from_dict(cls, parameters: Dict[str, Any]) -> "FieldConfig":
        """
        Initialize from a dictionary of values.

        Args:
            parameters: The parameter values.

        Returns:
            The initialized dataclass.

        """
        return cls(**parameters)


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
    row_num = plot_num // field_config.num_plots
    column_num = (field_config.num_plots - 1) - (
        plot_num % field_config.num_plots
    )

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
    all_results[DetectionColumns.SESSION.value] = session_name
    return all_results


def compute_heights(height_maps: Iterable[np.ndarray]) -> pd.DataFrame:
    """
    Computes the heights of each plot based on the elevation map for that plot.

    Args:
        height_maps: The plot elevation maps.

    Returns:
        A `DataFrame` with the columns in `HeightColumns`.

    """
    plot_heights = []
    for height_map in height_maps:
        # It uses negative values to signify invalid data, so we will just
        # remove those.
        height_map = np.reshape(height_map, (-1,))
        height_map = height_map[height_map >= 0]

        # Our height algorithm is very simple: We just take the highest point
        # and subtract the lowest point, assuming that the ground is locally
        # flat.
        plot_height = height_map.max() - height_map.min()
        plot_heights.append(plot_height)

    # Create the DataFrame.
    results = pd.DataFrame(
        data={
            HeightColumns.HEIGHT.value: plot_heights,
            HeightColumns.PLOT.value: range(len(plot_heights)),
        },
    )

    return results


def add_plot_index_for_heights(
    plot_heights: pd.DataFrame, field_config: FieldConfig
) -> pd.DataFrame:
    """
    Adds the correct field plot index to the height data.

    Args:
        plot_heights: The raw plot height data, with a detection plot number
            column.
        field_config: The field configuration to use.

    Returns:
        The same data, indexed by field plot number.

    """
    # Compute field plot numbers.
    plot_heights[HeightColumns.PLOT.value] = plot_heights[
        HeightColumns.PLOT.value
    ].apply(_to_field_plot_num, field_config=field_config)

    # Set the index.
    plot_heights.set_index(HeightColumns.PLOT.value, inplace=True)
    plot_heights.index.name = HeightColumns.PLOT.value
    plot_heights.sort_index(inplace=True)

    return plot_heights


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
    counts_per_plot[CountingColumns.PLOT.value] = counts_per_plot[
        CountingColumns.PLOT.value
    ].astype("uint64")
    # Use plot number as an index.
    counts_per_plot.set_index(CountingColumns.PLOT.value, inplace=True)
    counts_per_plot.index.name = CountingColumns.PLOT.value
    counts_per_plot.sort_index(inplace=True)

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


def create_per_plot_table(counting_results: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the counting results to a table with per-plot counts.

    Args:
        counting_results: The complete counting results.

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


def create_height_table(plot_heights: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a table containing human-readable height data.

    Args:
        plot_heights: The plot height data.

    Returns:
        A table containing plot numbers and corresponding heights.

    """
    # Make the headers a little more descriptive.
    plot_heights.rename(
        {HeightColumns.HEIGHT.value: "Height (m)"},
        axis="columns",
        inplace=True,
    )
    # Turn the plot index into a column so that it shows up in the spreadsheet.
    plot_heights.insert(0, "Plot", plot_heights.index.values)

    return plot_heights


def clean_genotypes(raw_genotypes: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans up the genotype data that was loaded from the genotype spreadsheet.

    Args:
        raw_genotypes: The raw genotype information that was loaded.

    Returns:
        The cleaned genotype table.

    """
    # Remove extraneous columns.
    needed_columns = {e.value for e in GenotypeColumns}
    cleaned = raw_genotypes.drop(
        columns=frozenset(raw_genotypes.columns) - needed_columns
    )

    # Index by plot number.
    cleaned.set_index(GenotypeColumns.PLOT.value, inplace=True)
    cleaned.sort_index(inplace=True)

    return cleaned


def clean_ground_truth(raw_ground_truth: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans up the ground truth data loaded from the spreadsheet.

    Args:
        raw_ground_truth: The raw ground-truth information that was loaded.

    Returns:
        The cleaned ground-truth table.

    """
    # Remove extraneous columns.
    needed_columns = {e.value for e in GroundTruthColumns}
    cleaned = raw_ground_truth.drop(
        columns=frozenset(raw_ground_truth.columns) - needed_columns
    )

    # Convert plot numbers to integers.
    cleaned[GroundTruthColumns.PLOT.value] = cleaned[
        GroundTruthColumns.PLOT.value
    ].astype("uint64")
    # Convert sessions to ISO strings.
    cleaned[GroundTruthColumns.SESSION.value] = cleaned[
        GroundTruthColumns.SESSION.value
    ].apply(lambda d: d.date().isoformat())

    # Remove counts that have NA in the manual count column. These are
    # generally for plots that have invalid data.
    cleaned.dropna(
        subset=GroundTruthColumns.MANUAL_IMAGE_COUNT.value, inplace=True
    )

    # Remove Jeevan's counts.
    cleaned = cleaned[cleaned[GroundTruthColumns.PERSON.value] != "JA"]

    # Index by plot number.
    cleaned.set_index(
        GroundTruthColumns.PLOT.value, inplace=True,
    )
    cleaned.sort_index(inplace=True)

    return cleaned


def find_empty_plots(
    *, cumulative_counts: pd.DataFrame, num_empty_plots: int
) -> List[int]:
    """
    Uses a very simple approach to find the plots that are probably empty.
    Basically, it just assumes that the plots with the lowest number of total
    flowers are empty.

    Args:
        cumulative_counts: The cumulative counting results.
        num_empty_plots: The total number of empty plots in the field.

    Returns:
        List of the plots that are probably empty.

    """
    # Keep only the final counts for each plot.
    only_counts = cumulative_counts[CountingColumns.COUNT.value]
    plot_groups = only_counts.groupby([only_counts.index])
    total_counts = plot_groups.agg("max")

    # Find the smallest ones.
    total_counts.sort_values(inplace=True)
    return total_counts.index.tolist()[:num_empty_plots]


def clean_empty_plots(
    *, plot_df: pd.DataFrame, empty_plots: List[int]
) -> pd.DataFrame:
    """
    Removes rows from a dataframe that correspond to empty plots.

    Args:
        plot_df: The dataframe to clean, indexed by plot number.
        empty_plots: The list of plots that are empty.

    Returns:
        The same dataframe, but without rows for empty plots.

    """
    plot_df.drop(empty_plots, inplace=True)
    return plot_df


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

    return pd.merge(
        counting_results, ground_truth, left_index=True, right_index=True
    )


def compute_flowering_peak(counting_results: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the peak flowering times for each plot.

    Args:
        counting_results: The complete counting results.

    Returns:
        A dataframe in the same format as the input, but containing only the
        rows that represent the peak flower count for each plot.

    """

    def _find_peak(plot_counts: pd.DataFrame) -> pd.DataFrame:
        plot_counts.sort_values(by=CountingColumns.DAP.value, inplace=True)
        return plot_counts.iloc[
            plot_counts[CountingColumns.COUNT.value].argmax()
        ]

    # Find the rows with the maximum count for each plot.
    plot_groups = counting_results.groupby([counting_results.index])
    return plot_groups.apply(_find_peak)


def compute_flowering_start_end(
    counting_results: pd.DataFrame,
    *,
    start_threshold: float,
    end_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes the flowering window for each plot.

    Args:
        counting_results: The complete counting results.
        start_threshold: The threshold at which to consider flowering started.
        end_threshold: The threshold at which to consider flowering to be over.

    Returns:
        Two dataframes in the same format as the input, but containing only the
        rows that represent the start and end of flowering for each plot,
        respectively.

    """
    counting_results.sort_values(by=CountingColumns.DAP.value, inplace=True)

    def _get_flowering_start(plot_counts: pd.DataFrame) -> pd.DataFrame:
        # Find the total number of flowers for this plot.
        cumulative_counts = plot_counts[CountingColumns.COUNT.value].cumsum()
        total_flowers = cumulative_counts.max()
        # Find the thresholds.
        start_flowers = total_flowers * start_threshold

        # Filter to the correct row.
        up_to_start = plot_counts[cumulative_counts <= start_flowers]
        if up_to_start.empty:
            # Edge case: start is as the beginning then.
            return plot_counts.iloc[0]
        return up_to_start.iloc[
            up_to_start[CountingColumns.DAP.value].argmax()
        ]

    def _get_flowering_end(plot_counts: pd.DataFrame) -> pd.DataFrame:
        # Find the total number of flowers for this plot.
        cumulative_counts = plot_counts[CountingColumns.COUNT.value].cumsum()
        total_flowers = cumulative_counts.max()
        # Find the thresholds.
        end_flowers = total_flowers * end_threshold

        # Filter to the correct row.
        end_and_after = plot_counts[cumulative_counts >= end_flowers]
        return end_and_after.iloc[
            end_and_after[CountingColumns.DAP.value].argmin()
        ]

    # Find the rows for each plot that signify the start of counting.
    plot_groups = counting_results.groupby([counting_results.index])
    return (
        plot_groups.parallel_apply(_get_flowering_start),
        plot_groups.parallel_apply(_get_flowering_end),
    )


def compute_flowering_duration(
    *, flowering_starts: pd.DataFrame, flowering_ends: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes the total flowering duration.

    Args:
        flowering_starts: The flowering start times.
        flowering_ends: The flowering end times.

    Returns:
        Dataframe containing the flowering durations.

    """
    flowering_starts.sort_index(inplace=True)
    flowering_ends.sort_index(inplace=True)

    durations = (
        flowering_ends[CountingColumns.DAP.value]
        - flowering_starts[CountingColumns.DAP.value]
    )
    return pd.DataFrame(data={FloweringTimeColumns.DURATION.value: durations})


def compute_cumulative_counts(counting_results: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cumulative flower counts for each plot.

    Args:
        counting_results: The complete counting results.

    Returns:
        The same counting results, but the "count" column is now cumulative.

    """

    def _count_cumulative(plot_counts: pd.DataFrame) -> pd.DataFrame:
        # Sort by DAP.
        plot_counts.sort_values(by=[CountingColumns.DAP.value], inplace=True)
        # Calculate cumulative counts.
        plot_counts[CountingColumns.COUNT.value] = plot_counts[
            CountingColumns.COUNT.value
        ].cumsum()

        return plot_counts

    plot_groups = counting_results.groupby(
        [counting_results.index], group_keys=False
    )
    return plot_groups.parallel_apply(_count_cumulative)


def compute_flowering_ramps(
    *,
    peak_flowering_times: pd.DataFrame,
    flowering_start_times: pd.DataFrame,
    cumulative_counts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Computes the slope of the initial ramp in flowering.

    Args:
        peak_flowering_times: The peak flowering time data.
        flowering_start_times: The flowering start time data.
        cumulative_counts: The complete counting results, with cumulative
            counts.

    Returns:
        A `DataFrame` indexed by plot with flowering ramp information.

    """

    def _fit_line_for_plot(plot_counts: pd.DataFrame) -> pd.Series:
        # All plots should be the same in this DF.
        plot_num = plot_counts.index[0]
        # Filter the counting results to only those between the flowering
        # start and end.
        start_dap = flowering_start_times.loc[plot_num][
            CountingColumns.DAP.value
        ]
        peak_dap = peak_flowering_times.loc[plot_num][
            CountingColumns.DAP.value
        ]
        plot_daps = plot_counts[CountingColumns.DAP.value]
        plot_counts_ramp = plot_counts[
            (plot_daps >= start_dap) & (plot_daps <= peak_dap)
        ]

        # Fit a line to the resulting counts.
        model = sm.ols(
            f"{CountingColumns.COUNT.value} ~ {CountingColumns.DAP.value}",
            data=plot_counts_ramp,
        ).fit()

        return pd.Series(
            {
                FloweringSlopeColumns.SLOPE.value: model.params[1],
                FloweringSlopeColumns.INTERCEPT.value: model.params[0],
            },
        )

    plot_groups = cumulative_counts.groupby([cumulative_counts.index])
    return plot_groups.parallel_apply(_fit_line_for_plot)


def _merge_genotype_info(
    *, flower_data: pd.DataFrame, genotypes: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges a dataframe indexed by plot with genotype information, filters out
    the extraneous genotypes, and averages all the replicates.

    Args:
        flower_data: The flowering data, indexed by plot number.
        genotypes: The dataframe containing genotype information.

    Returns:
        The merged data.

    """
    # Merge flowering and genotype data together for easy plotting.
    combined_data = pd.merge(
        flower_data, genotypes, left_index=True, right_index=True
    )
    # Ignore data from the extra populations, since they are too small
    # to do anything useful with.
    population = combined_data[GenotypeColumns.POPULATION.value]
    combined_data = combined_data[
        population.str.contains("Pima") | population.str.contains("Maxxa")
    ]

    # Average the replicates for each genotype together.
    group_columns = [
        GenotypeColumns.GENOTYPE.value,
        GenotypeColumns.POPULATION.value,
    ]
    if CountingColumns.DAP.value in combined_data.columns:
        # Group by DAP too if the data are temporal.
        group_columns.append(CountingColumns.DAP.value)
    return combined_data.groupby(group_columns, as_index=False,).agg("mean")


def _plot_flowering_time_histogram(
    flower_data: pd.DataFrame, *, genotypes: pd.DataFrame, title: str
) -> plot.Figure:
    """
    Draws a histogram of some flowering attribute, predicated on DAP and
    colored by population.

    Args:
        flower_data: The flowering data we want to plot. Should have one row
            for each plot and have a DAP column.
        genotypes: The dataframe containing genotype information.
        title: The title to use for the plot.

    Returns:
        The plot that it made.

    """
    combined_data = _merge_genotype_info(
        flower_data=flower_data, genotypes=genotypes
    )

    # Plot it.
    axes = sns.histplot(
        data=combined_data,
        x=CountingColumns.DAP.value,
        hue=GenotypeColumns.POPULATION.value,
        multiple="dodge",
        shrink=0.8,
        # Use one bin for each session.
        bins=len(flower_data[CountingColumns.SESSION.value].unique()),
    )
    axes.set_title(title)
    axes.set(xlabel="Days After Planting", ylabel="# of Genotypes")

    return plot.gcf()


def _plot_flowering_time_means(
    flower_data: pd.DataFrame, *, genotypes: pd.DataFrame, title: str
) -> plot.Figure:
    """
    Creates a bar plot that facilitates the analysis of whether there are
    significant differences between the flowering time attributes of each
    population.

    Args:
        flower_data: The flowering data we want to plot. Should have one row
            for each plot and have a DAP column.
        genotypes: The dataframe containing genotype information.
        title: The title to use for the plot.

    Returns:
        The plot that it made.

    """
    # Merge flowering and genotype data together for easy plotting.
    combined_data = pd.merge(
        flower_data, genotypes, left_index=True, right_index=True
    )

    # Plot it.
    axes = sns.barplot(
        x=GenotypeColumns.POPULATION.value,
        y=CountingColumns.DAP.value,
        data=combined_data,
        capsize=0.2,
    )
    axes.set_title(title)
    axes.set(xlabel="Population", ylabel="Days After Planting")

    figure = plot.gcf()
    # Make it wider so the x labels don't overlap.
    figure.set_size_inches(12, 6)
    return figure


def plot_flowering_slope_dist(
    *, flowering_slopes: pd.DataFrame, genotypes: pd.DataFrame
) -> plot.Figure:
    """
    Plots a histogram of the slopes of the initial flowering ramp-up for each
    genotype.

    Args:
        flowering_slopes: The extracted flowering slope data.
        genotypes: The dataframe containing genotype information.

    Returns:
        The plot that it created.

    """
    combined_data = _merge_genotype_info(
        flower_data=flowering_slopes, genotypes=genotypes
    )

    # Plot it.
    axes = sns.histplot(
        data=combined_data,
        x=FloweringSlopeColumns.SLOPE.value,
        hue=GenotypeColumns.POPULATION.value,
        multiple="dodge",
        shrink=0.8,
    )
    axes.set_title("Flowering Slope")
    axes.set(xlabel="Slope (flowers/day)", ylabel="# of Genotypes")

    return plot.gcf()


def plot_flowering_slope_comparison(
    *, flowering_slopes: pd.DataFrame, genotypes: pd.DataFrame
) -> plot.Figure:
    """
    Plots a comparison of the flowering slopes of different populations.

    Args:
        flowering_slopes: The extracted flowering slope data.
        genotypes: The dataframe containing genotype information.

    Returns:
        The plot that it created.

    """
    # Merge flowering and genotype data together for easy plotting.
    combined_data = pd.merge(
        flowering_slopes, genotypes, left_index=True, right_index=True
    )

    # Plot it.
    axes = sns.barplot(
        x=GenotypeColumns.POPULATION.value,
        y=FloweringSlopeColumns.SLOPE.value,
        data=combined_data,
        capsize=0.2,
    )
    axes.set_title("Mean Flowering Slope")
    axes.set(xlabel="Population", ylabel="Slope (flowers/day)")

    figure = plot.gcf()
    # Make it wider so the x labels don't overlap.
    figure.set_size_inches(12, 6)
    return figure


def plot_peak_flowering_dist(
    *, peak_flowering_times: pd.DataFrame, **kwargs: Any
) -> plot.Figure:
    """
    Plots a histogram of the peak flowering times.

    Args:
        peak_flowering_times: Dataset containing peak flowering times for
            each plot.
        **kwargs: Will be forwarded to `_plot_flowering_time_histogram`.

    Returns:
        The plot that it made.

    """
    return _plot_flowering_time_histogram(
        peak_flowering_times, **kwargs, title="Peak Flowering Time"
    )


def plot_peak_flowering_comparison(
    *, peak_flowering_times: pd.DataFrame, **kwargs: Any
) -> plot.Figure:
    """
    Plots a comparison of the peak flowering times of different populations.

    Args:
        peak_flowering_times: Dataset containing peak flowering times for
            each plot.
        **kwargs: Will be forwarded to `_plot_flowering_time_histogram`.

    Returns:
        The plot that it made.

    """
    return _plot_flowering_time_means(
        peak_flowering_times, **kwargs, title="Mean Peak Flowering Time"
    )


def plot_flowering_start_dist(
    *, flowering_start_times: pd.DataFrame, **kwargs: Any
) -> plot.Figure:
    """
    Plots a histogram of the flowering end times.

    Args:
        flowering_start_times: Dataset containing flowering start times for
            each plot.
        **kwargs: Will be forwarded to `_plot_flowering_time_histogram`.

    Returns:
        The plot that it made.

    """
    return _plot_flowering_time_histogram(
        flowering_start_times, **kwargs, title="Flowering Start Time"
    )


def plot_flowering_start_comparison(
    *, flowering_start_times: pd.DataFrame, **kwargs: Any
) -> plot.Figure:
    """
    Plots a comparison of the flowering start times of different populations.

    Args:
        flowering_start_times: Dataset containing flowering start times for
            each plot.
        **kwargs: Will be forwarded to `_plot_flowering_time_histogram`.

    Returns:
        The plot that it made.

    """
    return _plot_flowering_time_means(
        flowering_start_times, **kwargs, title="Mean Flowering Start Time"
    )


def plot_flowering_end_dist(
    *, flowering_end_times: pd.DataFrame, **kwargs: Any
) -> plot.Figure:
    """
    Plots a histogram of the flowering start times.

    Args:
        flowering_end_times: Dataset containing flowering start times for
            each plot.
        **kwargs: Will be forwarded to `_plot_flowering_time_histogram`.

    Returns:
        The plot that it made.

    """
    return _plot_flowering_time_histogram(
        flowering_end_times, **kwargs, title="Flowering End Time"
    )


def plot_flowering_end_comparison(
    *, flowering_end_times: pd.DataFrame, **kwargs: Any
) -> plot.Figure:
    """
    Plots a comparison of the flowering end times of different populations.

    Args:
        flowering_end_times: Dataset containing flowering end times for
            each plot.
        **kwargs: Will be forwarded to `_plot_flowering_time_histogram`.

    Returns:
        The plot that it made.

    """
    return _plot_flowering_time_means(
        flowering_end_times, **kwargs, title="Mean Flowering End Time"
    )


def plot_flowering_duration_dist(
    *, flowering_durations: pd.DataFrame, genotypes: pd.DataFrame
) -> plot.Figure:
    """
    Plots a histogram of the flowering start times.

    Args:
        flowering_durations: Dataset containing flowering durations for each
            plot.
        genotypes: The cleaned genotype information.

    Returns:
        The plot that it made.

    """
    combined_data = _merge_genotype_info(
        flower_data=flowering_durations, genotypes=genotypes
    )

    # Plot it.
    axes = sns.histplot(
        data=combined_data,
        x=FloweringTimeColumns.DURATION.value,
        hue=GenotypeColumns.POPULATION.value,
        multiple="dodge",
        shrink=0.8,
    )
    axes.set_title("Flowering Duration")
    axes.set(xlabel="Days", ylabel="# of Genotypes")

    return plot.gcf()


def plot_flowering_duration_comparison(
    *, flowering_durations: pd.DataFrame, genotypes: pd.DataFrame
) -> plot.Figure:
    """
    Plots a comparison of the flowering end times of different populations.

    Args:
        flowering_durations: Dataset containing flowering durations for
            each plot.
        genotypes: The cleaned genotype information.

    Returns:
        The plot that it made.

    """
    # Merge flowering and genotype data together for easy plotting.
    combined_data = pd.merge(
        flowering_durations, genotypes, left_index=True, right_index=True
    )

    # Plot it.
    axes = sns.barplot(
        x=GenotypeColumns.POPULATION.value,
        y=FloweringTimeColumns.DURATION.value,
        data=combined_data,
        capsize=0.2,
    )
    axes.set_title("Mean Flowering Duration")
    axes.set(xlabel="Population", ylabel="Days")

    figure = plot.gcf()
    # Make it wider so the x labels don't overlap.
    figure.set_size_inches(12, 6)
    return figure


def plot_height_dist(
    *, plot_heights: pd.DataFrame, genotypes: pd.DataFrame
) -> plot.Figure:
    """
    Plots a histogram of the plot heights.

    Args:
        plot_heights: Dataset containing heights for each plot.
        genotypes: The cleaned genotype information.

    Returns:
        The plot that it made.

    """
    combined_data = _merge_genotype_info(
        flower_data=plot_heights, genotypes=genotypes
    )

    # Plot it.
    axes = sns.histplot(
        data=combined_data,
        x=HeightColumns.HEIGHT.value,
        hue=GenotypeColumns.POPULATION.value,
        multiple="dodge",
        shrink=0.8,
    )
    axes.set_title("Final Plot Heights")
    axes.set(xlabel="Height (m)", ylabel="# of Genotypes")

    return plot.gcf()


def plot_height_comparison(
    *, plot_heights: pd.DataFrame, genotypes: pd.DataFrame
) -> plot.Figure:
    """
    Plots a comparison of the flowering end times of different populations.

    Args:
        plot_heights: Dataset containing heights for each plot.
        genotypes: The cleaned genotype information.

    Returns:
        The plot that it made.

    """
    # Merge flowering and genotype data together for easy plotting.
    combined_data = pd.merge(
        plot_heights, genotypes, left_index=True, right_index=True
    )

    # Plot it.
    axes = sns.barplot(
        x=GenotypeColumns.POPULATION.value,
        y=HeightColumns.HEIGHT.value,
        data=combined_data,
        capsize=0.2,
    )
    axes.set_title("Mean Final Plot Heights")
    axes.set(xlabel="Population", ylabel="Height (m)")

    figure = plot.gcf()
    # Make it wider so the x labels don't overlap.
    figure.set_size_inches(12, 6)
    return figure


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


def plot_ground_truth_vs_predicted(
    *, counts_with_gt: pd.DataFrame, genotypes: pd.DataFrame
) -> plot.Figure:
    """
    Creates a scatter plot showing the ground-truth vs. predicted counts.

    Args:
        counts_with_gt: The merged counting results and ground-truth data.
        genotypes: The cleaned genotype information.

    Returns:
        The plot that it created.

    """
    # Merge flowering and genotype data together for easy plotting.
    counts_with_gt.reset_index(
        level=GroundTruthColumns.DAP.value, inplace=True
    )
    combined_data = pd.merge(
        counts_with_gt, genotypes, left_index=True, right_index=True
    )

    # Plot the regression.
    axes = sns.scatterplot(
        data=combined_data,
        x=GroundTruthColumns.TRUE_COUNT.value,
        y=CountingColumns.COUNT.value,
        hue=CountingColumns.DAP.value,
        style=GenotypeColumns.POPULATION.value,
    )
    axes.set_title("Predicted vs. Ground-Truth Counts")
    axes.set(xlabel="Ground-Truth", ylabel="Automatic")

    return plot.gcf()


def plot_flowering_curves(
    *, cumulative_counts: pd.DataFrame, genotypes: pd.DataFrame
) -> Iterable[plot.Figure]:
    """
    Creates a flowering curve for each individual plot.

    Args:
        cumulative_counts: The complete counting results, with cumulative
            counts.
        genotypes: The cleaned genotype information.

    Yields:
        The curves for each plot, in order.

    """
    # Get all the sessions to use as a common index.
    all_daps = cumulative_counts[CountingColumns.DAP.value].unique()
    all_daps.sort()

    # Merge flowering and genotype data together for easy plotting.
    combined_data = pd.merge(
        cumulative_counts, genotypes, left_index=True, right_index=True
    )
    combined_data.index.name = CountingColumns.PLOT.value
    # Drop columns we don't need to save memory.
    genotype_columns = {e.value for e in GenotypeColumns}
    counting_columns = {e.value for e in CountingColumns}
    combined_data.drop(
        columns=genotype_columns
        | counting_columns
        - {
            CountingColumns.DAP.value,
            CountingColumns.COUNT.value,
            GenotypeColumns.GENOTYPE.value,
        },
        errors="ignore",
    )

    # Re-index by DAP.
    combined_data.set_index(
        CountingColumns.DAP.value, append=True, inplace=True
    )
    combined_data.sort_index(inplace=True)

    def _add_missing_sessions(plot_counts: pd.DataFrame) -> pd.DataFrame:
        plot_genotype = plot_counts[GenotypeColumns.GENOTYPE.value].iloc[0]
        # Expand the index with any sessions we don't have data for.
        plot_counts.reset_index(
            level=CountingColumns.PLOT.value, inplace=True, drop=True
        )
        new_index = pd.Index(all_daps, name=CountingColumns.DAP.value)
        plot_counts = plot_counts.reindex(
            new_index, method="ffill", fill_value=0
        )
        # Keep the genotype consistent, in case it was filled with zeros.
        plot_counts[GenotypeColumns.GENOTYPE.value] = plot_genotype
        return plot_counts

    # Group by plot, which is now the first level of the index.
    plot_groups = combined_data.groupby(lambda row: row[0])
    combined_data = plot_groups.parallel_apply(_add_missing_sessions)
    combined_data.index.set_names(
        [CountingColumns.PLOT.value, CountingColumns.DAP.value], inplace=True
    )

    # Group by genotype, and plot all replicates on the same axes.
    genotype_groups = combined_data.groupby([GenotypeColumns.GENOTYPE.value])

    for genotype, genotype_indices in genotype_groups.indices.items():
        # Get the rows pertaining to this genotype.
        genotype_rows = combined_data.iloc[genotype_indices]

        # Extract the index as columns, so we can plot it.
        genotype_rows.reset_index(inplace=True)
        # Plot the curve.
        axes = sns.lineplot(
            data=genotype_rows,
            x=CountingColumns.DAP.value,
            y=CountingColumns.COUNT.value,
            hue=CountingColumns.PLOT.value,
        )
        axes.set_title(f"Genotype {genotype}")
        axes.set(
            xlabel="Days After Planting", ylabel="Cumulative # of Flowers"
        )

        yield plot.gcf()


def plot_mean_flowering_curve(
    *, cumulative_counts: pd.DataFrame, genotypes: pd.DataFrame
) -> plot.Figure:
    """
    Creates mean flowering curves for each population.

    Args:
        cumulative_counts: The complete counting results, with cumulative
            counts.
        genotypes: The cleaned genotype information.

    Returns:
        The plot of the flowering curves.

    """
    # Merge flowering and genotype data together for easy plotting.
    combined_data = _merge_genotype_info(
        flower_data=cumulative_counts, genotypes=genotypes
    )

    # Plot the curve.
    axes = sns.lineplot(
        data=combined_data,
        x=CountingColumns.DAP.value,
        y=CountingColumns.COUNT.value,
        hue=GenotypeColumns.POPULATION.value,
    )
    axes.set_title("Average Flowering Curves")
    axes.set(xlabel="Days After Planting", ylabel="Cumulative # of Flowers")

    return plot.gcf()
