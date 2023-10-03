"""
Common functionality between the two versions of the pipeline.
"""
import enum
from datetime import date
from functools import partial
from typing import Dict, Iterable, List, Tuple, Set, Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plot
from statsmodels.formula import api as sm


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


@enum.unique
class OutlierColumns(enum.Enum):
    """
    Names of columns for the outlier info DF.
    """

    TYPE = "outlier_type"
    """
    The type of outlier See the `Outlier` enum for the possible values.
    """

    START = "start_outlier"
    END = "end_outlier"
    DURATION = "duration_outlier"
    PEAK = "peak_outlier"
    SLOPE = "slope_outlier"
    """
    Outlier types for specific metrics.
    """


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

    SAMPLE = "sample"
    """
    The sample number when using Monte Carlo sampling on plot locations.
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
    subset = [DetectionColumns.SESSION.value, DetectionColumns.PLOT_NUM.value]
    if DetectionColumns.SAMPLE.value in detection_results.columns:
        # Count within each sample. We will merge them later.
        subset.append(DetectionColumns.SAMPLE.value)
    counts_per_plot_series = detection_results.value_counts(subset=subset)
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


def choose_best_counts(counts_with_gt: pd.DataFrame) -> pd.DataFrame:
    """
    For cases where we have multiple possible counts for each plot,
    this chooses the ones that are closest to the GT for each plot.

    Args:
        counts_with_gt: The merged counting results and ground-truth data.

    Returns:
        The merged results.

    """

    def _find_best(row: pd.Series) -> Tuple[int, float]:
        count_error = np.abs(
            row[CountingColumns.COUNT.value]
            - row[GroundTruthColumns.TRUE_COUNT.value]
        )
        best_error_idx = count_error.argmin()
        return row[CountingColumns.COUNT.value].iloc[best_error_idx]

    best_counts = counts_with_gt.groupby(
        level=[CountingColumns.PLOT.value, CountingColumns.DAP.value]
    ).parallel_apply(_find_best)

    # Update the dataframe, removing the additional samples.
    counts_with_gt = counts_with_gt[
        counts_with_gt[DetectionColumns.SAMPLE.value] == 0
    ].copy()
    counts_with_gt[CountingColumns.COUNT.value] = best_counts
    return counts_with_gt


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
    plot_groups = counting_results.groupby(counting_results.index)
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
    if len(counting_results) > 1:
        return (
            plot_groups.parallel_apply(_get_flowering_start),
            plot_groups.parallel_apply(_get_flowering_end),
        )
    else:
        return (
            plot_groups.apply(_get_flowering_start),
            plot_groups.apply(_get_flowering_end),
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
        counting_results.index, group_keys=False
    )
    if len(counting_results) > 1:
        return plot_groups.parallel_apply(_count_cumulative)
    else:
        # Work around a bug in Pandarallel.
        return plot_groups.apply(_count_cumulative)


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
    if len(cumulative_counts) > 1:
        return plot_groups.parallel_apply(_fit_line_for_plot)
    else:
        return plot_groups.apply(_fit_line_for_plot)


def create_metric_table(
    *,
    peak_flowering_times: pd.DataFrame,
    flowering_starts: pd.DataFrame,
    flowering_ends: pd.DataFrame,
    flowering_durations: pd.DataFrame,
    flowering_slopes: pd.DataFrame,
    genotypes: pd.DataFrame,
    outliers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Combines all the per-plot metrics into a single, human-readable table.

    Args:
        peak_flowering_times: The peak flowering times.
        flowering_starts: The flowering start times.
        flowering_ends: The flowering end times.
        flowering_durations: The flowering durations.
        flowering_slopes: The flowering slopes.
        genotypes: The genotype information.
        outliers: Optional outlier information, which will also be included
            if present.

    Returns:
        The combined table with all metrics.

    """
    # Everything should have the same index, so merge it all together.
    merge = partial(pd.merge, left_index=True, right_index=True)
    peak_and_start = merge(
        peak_flowering_times, flowering_starts, suffixes=("_peak", "_start")
    )
    end_and_duration = merge(flowering_ends, flowering_durations)
    slope_and_genotype = merge(flowering_slopes, genotypes)
    combined = merge(peak_and_start, end_and_duration)
    combined = merge(combined, slope_and_genotype)

    # Average across genotypes.
    combined = combined.groupby(
        GenotypeColumns.GENOTYPE.value, as_index=False
    ).mean()
    combined.set_index(GenotypeColumns.GENOTYPE.value, inplace=True)

    if outliers is not None:
        combined = merge(combined, outliers)

    # Convert to human-readable names.
    combined.rename(
        columns={
            f"{CountingColumns.SESSION.value}_peak": "Peak Session",
            f"{CountingColumns.DAP.value}_peak": "Peak DAP",
            f"{CountingColumns.COUNT.value}_peak": "Peak Count",
            f"{CountingColumns.SESSION.value}_start": "Start Session",
            f"{CountingColumns.DAP.value}_start": "Start DAP",
            f"{CountingColumns.COUNT.value}_start": "Start Count",
            CountingColumns.SESSION.value: "End Session",
            CountingColumns.DAP.value: "End DAP",
            CountingColumns.COUNT.value: "End Count",
            FloweringTimeColumns.DURATION.value: "Duration (days)",
            FloweringSlopeColumns.SLOPE.value: "Slope (flowers/day)",
            FloweringSlopeColumns.INTERCEPT.value: "Intercept",
            GenotypeColumns.GENOTYPE.value: "Genotype",
            GenotypeColumns.POPULATION.value: "Population",
            OutlierColumns.START.value: "Start Outlier",
            OutlierColumns.END.value: "End Outlier",
            OutlierColumns.DURATION.value: "Duration Outlier",
            OutlierColumns.PEAK.value: "Peak Outlier",
            OutlierColumns.SLOPE.value: "Slope Outlier",
        },
        inplace=True,
    )
    # Turn the index into a column so it shows up in the spreadsheet.
    combined.insert(0, "Genotype", combined.index.values)

    return combined


def merge_genotype_info(
    *,
    flower_data: pd.DataFrame,
    genotypes: pd.DataFrame,
    outliers: pd.DataFrame | None = None,
    group_on_dap: bool = False,
    filter_populations: Set[str] | None = None,
) -> pd.DataFrame:
    """
    Merges a dataframe indexed by plot with genotype information, filters out
    the extraneous genotypes, and averages all the replicates.

    Args:
        flower_data: The flowering data, indexed by plot number.
        genotypes: The dataframe containing genotype information.
        outliers: The outlier data. If provided, it will merge this too.
        group_on_dap: If true, will also group by DAP, instead of averaging
            DAPs.
        filter_populations: If specified, will only keep data from these
            specific populations.

    Returns:
        The merged data.

    """
    # Merge flowering and genotype data together for easy plotting.
    combined_data = pd.merge(
        flower_data, genotypes, left_index=True, right_index=True
    )

    if filter_populations is not None:
        # Remove extraneous populations.
        for population in filter_populations:
            combined_data = combined_data[
                combined_data[GenotypeColumns.POPULATION.value].str.contains(
                    population
                )
            ]

    # Average the replicates for each genotype together.
    group_columns = [
        GenotypeColumns.GENOTYPE.value,
        GenotypeColumns.POPULATION.value,
    ]
    if group_on_dap:
        # Group by DAP too if the data are temporal.
        group_columns.append(CountingColumns.DAP.value)
    combined_data = combined_data.groupby(
        group_columns,
        as_index=False,
    ).agg("mean")

    if outliers is not None:
        combined_data.set_index(GenotypeColumns.GENOTYPE.value, inplace=True)
        combined_data = pd.merge(
            combined_data, outliers, left_index=True, right_index=True
        )
    return combined_data


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
    combined_data = merge_genotype_info(
        flower_data=cumulative_counts, genotypes=genotypes, group_on_dap=True
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
    combined_data = merge_genotype_info(
        flower_data=flower_data,
        genotypes=genotypes,
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
        stat="density",
        common_norm=False,
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
    combined_data = merge_genotype_info(
        flower_data=flower_data, genotypes=genotypes
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
    combined_data = merge_genotype_info(
        flower_data=flowering_slopes,
        genotypes=genotypes,
    )

    # Plot it.
    axes = sns.histplot(
        data=combined_data,
        x=FloweringSlopeColumns.SLOPE.value,
        hue=GenotypeColumns.POPULATION.value,
        multiple="dodge",
        shrink=0.8,
        stat="density",
        common_norm=False,
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
    combined_data = merge_genotype_info(
        flower_data=flowering_durations,
        genotypes=genotypes,
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
