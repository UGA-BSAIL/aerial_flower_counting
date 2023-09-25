"""
Contains nodes for the `count_plots` pipeline.
"""

import enum
import re
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import torch.cuda
from loguru import logger
from matplotlib import pyplot as plot
from pandarallel import pandarallel
from PIL import Image
from pydantic.dataclasses import dataclass
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO

from ..common import (
    CountingColumns,
    DetectionColumns,
    GroundTruthColumns,
    batch_iter, FloweringSlopeColumns, FloweringTimeColumns,
)
from .visualization import draw_detections

_PREDICTION_BATCH_SIZE = 50
"""
Size of batches to use for prediction. This mostly impacts memory use.
"""

# Regular expression to match plot keys.
_PLOT_RE = re.compile(r"plot_(\d+)")
# Regular expression to match qualitative example plot image names.
_QUAL_EXAMPLE_PLOT_RE = re.compile(r"(\d+-\d+-\d+)_plot_(\d+)_(.*)")

sns.set_theme(
    context="paper", style="whitegrid", palette="husl", rc={"savefig.dpi": 600}
)

pandarallel.initialize()


@enum.unique
class HeightGtColumns(enum.Enum):
    """
    Names of the columns in the height ground-truth table.
    """

    PLOT = "Plot Number"
    """
    The field plot number.
    """
    PLANT_1_HEIGHT = "Plant 1 ht"
    """
    The height of the first plant.
    """
    PLANT_2_HEIGHT = "Plant 2 ht"
    """
    The height of the second plant.
    """
    PLANT_3_HEIGHT = "Plant 3 ht"
    """
    The height of the third plant.
    """
    MAX_HEIGHT = "mean_height"
    """
    The max height of the plants.
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


@enum.unique
class HeightColumns(enum.Enum):
    """
    Names of the columns in the plot height dataframe.
    """

    DETECTION_PLOT = DetectionColumns.PLOT_NUM.value
    """
    The detection plot number that these heights are from.
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
class FlowerSizeColumns(enum.Enum):
    """
    Names of the columns in the flower size dataframe.
    """

    SIZE_CM = "size_cm"
    """
    The size of the flower, in cm^2.
    """


ImagePartitionType = Dict[str, Callable[[], Image.Image]]
"""
Type of a partitioned dataset of images.
"""


@dataclass
class FieldConfig:
    """
    Represents the configuration of the actual field.

    Attributes:
        num_plots: The total number of plots in each row.

        first_row_num: The number assigned to the first row.
        first_plot_num: The number assigned to the first plot in each row.
        start_at_bottom: Whether the plot numbering starts at the bottom of the
            field. Otherwise, it's assumed it starts at the top.

        empty_rows: Set of rows that are not planted.
    """

    num_plots: int

    first_row_num: int
    first_plot_num: int
    start_at_bottom: bool

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
    if field_config.start_at_bottom:
        # Large numbers should be at the top.
        column_num = (field_config.num_plots - 1) - (
            plot_num % field_config.num_plots
        )
    else:
        column_num = plot_num % field_config.num_plots

    # Assign the correct row and plot numbers.
    row_num += field_config.first_row_num
    column_num += field_config.first_plot_num

    if row_num in field_config.empty_rows:
        # This row is not planted.
        return None

    # Create the complete plot number.
    return int(f"{row_num:02}{column_num:02}")


def detect_flowers(
    images: ImagePartitionType,
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
    model = YOLO(weights_file)

    # Infer on batches.
    results = []
    for batch_keys in batch_iter(
        images.keys(), batch_size=_PREDICTION_BATCH_SIZE
    ):
        logger.debug("Predicting on batch...")
        batch = [images[key]().convert("RGB") for key in batch_keys]
        batch_results = model.predict(batch, imgsz=max(image_size))

        # Add a new column indicating the source image.
        for key, image_results in zip(batch_keys, batch_results):
            results_df = pd.DataFrame(
                data=image_results.boxes.xyxy.cpu().numpy(),
                columns=[
                    DetectionColumns.X1.value,
                    DetectionColumns.Y1.value,
                    DetectionColumns.X2.value,
                    DetectionColumns.Y2.value,
                ],
            )
            results_df[
                DetectionColumns.CONFIDENCE.value
            ] = image_results.boxes.conf.cpu().numpy()
            results_df[DetectionColumns.BOX_NUM.value] = np.arange(
                len(results_df)
            )

            plot_num = int(_PLOT_RE.match(key).group(1))
            results_df[DetectionColumns.PLOT_NUM.value] = plot_num

            results.append(results_df)

    all_results = pd.concat(results, ignore_index=True)
    # Add a column for the session name.
    all_results[DetectionColumns.SESSION.value] = session_name
    return all_results


def load_sam_model(*, model_type: str, weights_file: str) -> SamPredictor:
    """
    Loads the SAM model to use for segmentation.

    Args:
        model_type: The model type to load.
        weights_file: The checkpoint file to load weights from.

    Returns:
        The loaded `SamPredictor`.

    """
    logger.debug("Loading SAM weights from {}...", weights_file)
    sam = sam_model_registry[model_type](
        checkpoint=Path(weights_file).as_posix()
    )
    if torch.cuda.is_available():
        sam = sam.cuda().eval()
    else:
        logger.warning(
            "Not using CUDA for SAM, because torch couldn't find it."
        )
        sam = sam.eval()
    return SamPredictor(sam)


def segment_flowers(
    images: ImagePartitionType,
    *,
    detections: pd.DataFrame,
    segmentor: SamPredictor,
    session_name: str,
) -> Dict[str, Callable[[], Image.Image]]:
    """
    Segments the flowers in a series of input images.

    Args:
        images: The images to segment flowers in.
        detections: The detection bounding boxes to use as segmentation prompts.
        segmentor: The model to use for segmentation.
        session_name: The name of this session.

    Returns:
        A large array with the segmentations for all the images, indexed by
        detection plot number and box number.

    """
    # Filter to only this session.
    detections = detections[
        detections[DetectionColumns.SESSION.value] == session_name
    ]
    by_plot = detections.set_index(DetectionColumns.PLOT_NUM.value)

    # Segments a particular bounding box.
    def _segment_box(
        image_: Callable[[], np.array], row_: pd.Series
    ) -> Image.Image:
        logger.debug("Segmenting plot...")
        segmentor.set_image(np.array(image_().convert("RGB")))

        box = row_[
            [
                DetectionColumns.X1.value,
                DetectionColumns.Y1.value,
                DetectionColumns.X2.value,
                DetectionColumns.Y2.value,
            ]
        ]
        mask, _, _ = segmentor.predict(
            box=box.to_numpy(), multimask_output=False
        )
        # Combine all the masks into one.
        return Image.fromarray(mask[0].astype(np.uint8) * 255, mode="L")

    partitions = {}
    for key, image in images.items():
        plot_num = int(_PLOT_RE.match(key).group(1))
        try:
            boxes = by_plot.loc[[plot_num]]
        except KeyError:
            # No detections for this image.
            continue

        for row_i, (_, row) in enumerate(boxes.iterrows()):
            session = row[DetectionColumns.SESSION.value]
            partitions[f"{session}_plot{plot_num:04}_box{row_i:02}"] = partial(
                _segment_box, image, row
            )
    return partitions


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
            HeightColumns.DETECTION_PLOT.value: range(len(plot_heights)),
        },
    )

    return results


def add_plot_index(
    plot_data: pd.DataFrame, field_config: FieldConfig
) -> pd.DataFrame:
    """
    Adds the correct field plot index to data that have only detection plots.

    Args:
        plot_data: The raw plot data, with a detection plot number
            column.
        field_config: The field configuration to use.

    Returns:
        The same data, indexed by field plot number.

    """
    # Compute field plot numbers.
    plot_data[CountingColumns.PLOT.value] = plot_data[
        DetectionColumns.PLOT_NUM.value
    ].apply(_to_field_plot_num, field_config=field_config)

    # Remove NaN values, because these are for empty plots.
    plot_data.dropna(inplace=True)
    plot_data[CountingColumns.PLOT.value] = plot_data[
        CountingColumns.PLOT.value
    ].astype("uint64")

    # Set the index.
    plot_data = plot_data.set_index(CountingColumns.PLOT.value)
    plot_data.index.name = CountingColumns.PLOT.value
    plot_data.sort_index(inplace=True)

    return plot_data


def collect_plot_heights(*plot_heights: pd.DataFrame) -> pd.DataFrame:
    """
    Collects the plot height results from multiple fields into a single
    Pandas dataframe. It expects them to have non-overlapping indices.

    Args:
        *plot_heights: The plot heights from each field.

    Returns:
        The combined plot heights.

    """
    return pd.concat(plot_heights)


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


def create_metric_table(
    *,
    peak_flowering_times: pd.DataFrame,
    flowering_starts: pd.DataFrame,
    flowering_ends: pd.DataFrame,
    flowering_durations: pd.DataFrame,
    flowering_slopes: pd.DataFrame,
    genotypes: pd.DataFrame,
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
        },
        inplace=True,
    )
    # Turn the index into a column so it shows up in the spreadsheet.
    combined.insert(0, "Plot", combined.index.values)

    return combined


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
        GroundTruthColumns.PLOT.value,
        inplace=True,
    )
    cleaned.sort_index(inplace=True)

    return cleaned


def clean_height_ground_truth(raw_ground_truth: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans up the height ground truth data and adds plot maximums.

    Args:
        raw_ground_truth: The raw ground-truth height information that was
            loaded.

    Returns:
        The cleaned ground-truth table.

    """
    # Remove the extra header row.
    first_col_name = raw_ground_truth.columns[0]
    # The extra header row is the only row with a value in the first column.
    cleaned = raw_ground_truth[raw_ground_truth[first_col_name].isnull()]

    # Drop the first column, since it's just the field name.
    cleaned = cleaned.drop(columns=first_col_name)
    # Remove "no plant" indicators.
    cleaned = cleaned.apply(pd.to_numeric, errors="coerce")

    # Convert to meters.
    cleaned[HeightGtColumns.PLANT_1_HEIGHT.value] *= 0.01
    cleaned[HeightGtColumns.PLANT_2_HEIGHT.value] *= 0.01
    cleaned[HeightGtColumns.PLANT_3_HEIGHT.value] *= 0.01

    # Add the max column.
    height_columns = cleaned.drop(columns=HeightGtColumns.PLOT.value)
    average_height = height_columns.max(axis=1, numeric_only=True)
    cleaned[HeightGtColumns.MAX_HEIGHT.value] = average_height

    # Index by plot number.
    cleaned.set_index(HeightGtColumns.PLOT.value, inplace=True)
    cleaned.sort_index(inplace=True)

    return cleaned


def clean_empty_plots(
    *,
    plot_df: pd.DataFrame,
    empty_plots: pd.DataFrame,
    field_config: FieldConfig,
) -> pd.DataFrame:
    """
    Removes rows from a dataframe that correspond to empty plots.

    Args:
        plot_df: The dataframe to clean, indexed by plot number.
        empty_plots: The list of (detection) plots that are empty.
        field_config: The configuration for this field.

    Returns:
        The same dataframe, but without rows for empty plots.

    """
    empty_plots = empty_plots.to_numpy()[:, 0]
    empty_plots = [
        _to_field_plot_num(p, field_config=field_config) for p in empty_plots
    ]
    # Remove Nones that result from rows that are not planted.
    empty_plots = [p for p in empty_plots if p is not None]

    plot_df.drop(empty_plots, inplace=True, errors="ignore")
    return plot_df


def merge_height_ground_truth(
    *, plot_heights: pd.DataFrame, ground_truth_heights: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges the ground-truth heights and estimated heights into a single
    `DataFrame` for easy comparison.

    Args:
        plot_heights: The estimated plot heights.
        ground_truth_heights: The ground-truth plot heights.

    Returns:
        The merged data, indexed by plot.

    """
    # Make sure the indices are named the same.
    ground_truth_heights.index.name = plot_heights.index.name
    # Merge the data for easy plotting.
    return pd.merge(
        plot_heights, ground_truth_heights, left_index=True, right_index=True
    )


def compute_flower_sizes(
    *,
    flower_masks: ImagePartitionType,
    detection_results: pd.DataFrame,
    gsds: Dict[str, float],
) -> pd.DataFrame:
    """
    Computes the average flower size for each plot.

    Args:
        flower_masks: The flower masks for each plot.
        detection_results: The detection results. These will be used to
            determine what plot masks we look at.
        gsds: A mapping of session names to GSD values for that session,
            in cm/px.

    Returns:
        A dataframe containing the average flower size for each plot.

    """

    def _compute_size(detection_row: pd.Series) -> float:
        """
        Computes the average flower size for a single detection.

        Args:
            detection_row: The row of the detection results for the plot.

        Returns:
            The average flower size for the plot.

        """
        session = detection_row[DetectionColumns.SESSION.value]
        plot_num = detection_row[DetectionColumns.PLOT_NUM.value]
        box_num = detection_row[DetectionColumns.BOX_NUM.value]

        # Compute the size based on the mask.
        try:
            mask_image = np.array(
                flower_masks[f"{session}_plot{plot_num:04}_box{box_num:02}"]()
            )
        except KeyError as error:
            logger.warning(f"Could not find mask for detection: {error}")
            return np.nan
        size_px = np.count_nonzero(mask_image)
        size_cm_2 = size_px * (gsds[session] * gsds[session])

        return size_cm_2

    detection_results[
        FlowerSizeColumns.SIZE_CM.value
    ] = detection_results.parallel_apply(_compute_size, axis=1)
    return detection_results.dropna()


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
    return combined_data.groupby(
        group_columns,
        as_index=False,
    ).agg("mean")


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


def plot_flower_size_comparison(
    *, flower_sizes: pd.DataFrame, genotypes: pd.DataFrame
) -> plot.Figure:
    """
    Plots a comparison of the flowering sizes of different populations.

    Args:
        flower_sizes: Dataset containing flower sizes for each plot.
        genotypes: The genotype data.

    Returns:
        The plot that it made.

    """
    # Merge flowering and genotype data together for easy plotting.
    combined_data = pd.merge(
        flower_sizes, genotypes, left_index=True, right_index=True
    )

    # Plot it.
    axes = sns.barplot(
        x=GenotypeColumns.POPULATION.value,
        y=FlowerSizeColumns.SIZE_CM.value,
        data=combined_data,
        capsize=0.2,
    )
    axes.set_title("Flower Size")
    axes.set(xlabel="Population", ylabel="Size (cm^2)")

    figure = plot.gcf()
    # Make it wider so the x labels don't overlap.
    figure.set_size_inches(12, 6)
    return figure


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


def plot_height_ground_truth_regression(
    heights_with_gt: pd.DataFrame,
) -> plot.Figure:
    """
    Plots the regression between the height estimation results and the
    ground-truth heights.

    Args:
        heights_with_gt: The merged height results and ground-truth data.

    Returns:
        The plot that it created.

    """
    # Plot the regression.
    axes = sns.lmplot(
        x=HeightColumns.HEIGHT.value,
        y=HeightGtColumns.MAX_HEIGHT.value,
        data=heights_with_gt,
    )
    axes.fig.suptitle("Estimated Height vs. Measured")
    axes.set_axis_labels(
        x_var="Estimated Height (m)", y_var="Measured Height (m)"
    )

    # Add R^2 value.
    def annotate(data: pd.DataFrame, **_: Any) -> None:
        r, p = stats.pearsonr(
            data[HeightColumns.HEIGHT.value],
            data[HeightGtColumns.MAX_HEIGHT.value],
        )
        global_axes = plot.gca()
        global_axes.text(
            0.05,
            0.8,
            f"r^2={r ** 2:.2f}, p={p:.2g}",
            transform=global_axes.transAxes,
        )

    axes.map_dataframe(annotate)

    return plot.gcf()


def plot_flowering_curves(
    *, cumulative_counts: pd.DataFrame, genotypes: pd.DataFrame
) -> Dict[str, Callable[[], plot.Figure]]:
    """
    Creates a flowering curve for each individual plot.

    Args:
        cumulative_counts: The complete counting results, with cumulative
            counts.
        genotypes: The cleaned genotype information.

    Returns:
        Partitioned dataset containing the figure for each genotype.

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

    def _plot_genotype(
        genotype_: str, genotype_indices_: List[int]
    ) -> plot.Figure:
        # Get the rows pertaining to this genotype.
        genotype_rows = combined_data.iloc[genotype_indices_]

        # Extract the index as columns, so we can plot it.
        genotype_rows.reset_index(inplace=True)
        # Plot the curve.
        axes = sns.lineplot(
            data=genotype_rows,
            x=CountingColumns.DAP.value,
            y=CountingColumns.COUNT.value,
            hue=CountingColumns.PLOT.value,
            palette="husl",
            linewidth=1.5,
        )
        axes.set_title(f"Genotype {genotype_}")
        axes.set(
            xlabel="Days After Planting", ylabel="Cumulative # of Flowers"
        )

        return plot.gcf()

    partitions = {}
    for genotype, genotype_indices in genotype_groups.indices.items():
        partitions[f"genotype_{genotype}"] = partial(
            _plot_genotype, genotype, genotype_indices
        )
    return partitions


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


def draw_qualitative_results(
    *,
    qualitative_examples: ImagePartitionType,
    detection_results: pd.DataFrame,
    flower_masks: ImagePartitionType,
) -> Dict[str, Image.Image]:
    """
    Draws the detection results for qualitative example plot images.

    Args:
        qualitative_examples: The qualitative example plot images.
        detection_results: The detection results.
        flower_masks: The flower segmentation masks.

    Returns:
        The same qualitative example images, with the results drawn on top.

    """
    detection_results = detection_results.set_index(
        DetectionColumns.PLOT_NUM.value
    )

    partitions = {}
    for example_name, example_image in qualitative_examples.items():
        # Extract the plot information.
        match = _QUAL_EXAMPLE_PLOT_RE.match(example_name)
        session = match.group(1)
        plot_number = int(match.group(2))

        # Get the example's detection results.
        example_detection_results = detection_results.loc[plot_number]
        example_detection_results = example_detection_results[
            example_detection_results[DetectionColumns.SESSION.value]
            == session
        ]

        # Get the masks for the detections.
        masks = []
        for _, detection in example_detection_results.iterrows():
            # Get the mask for this detection.
            box_num = detection[DetectionColumns.BOX_NUM.value]
            mask_key = f"{session}_plot{plot_number:04}_box{box_num:02}"
            masks.append(flower_masks[mask_key]())

        # Do the drawing.
        detections = example_detection_results[
            [
                DetectionColumns.X1.value,
                DetectionColumns.Y1.value,
                DetectionColumns.X2.value,
                DetectionColumns.Y2.value,
                DetectionColumns.CONFIDENCE.value,
            ]
        ]
        overlaid_image = draw_detections(
            example_image(), detections=detections.to_numpy(), masks=masks
        )
        partitions[example_name] = overlaid_image

    return partitions


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
