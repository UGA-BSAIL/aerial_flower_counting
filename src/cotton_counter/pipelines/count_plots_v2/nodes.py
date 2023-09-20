"""
Contains nodes for the updated plot counting pipeline.
"""

import enum
from functools import partial, reduce, cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from fiona.model import Feature, Geometry, Properties
from loguru import logger
from matplotlib import pyplot as plot
from pandarallel import pandarallel
from PIL import Image
from shapely import Polygon, STRtree, intersection
from shapely.geometry import mapping
from shapely import affinity, from_ragged_array, GeometryType
from ultralytics import YOLO

from .camera_utils import CameraConfig, CameraTransformer, MissingImageError
from ..common import (
    CountingColumns,
    DetectionColumns,
    GroundTruthColumns,
    batch_iter,
)
from .field_config import FieldConfig
from rasterio import DatasetReader

ImageDataSet = Dict[str, Callable[[], Image.Image]]
"""
Type alias for a dataset containing multiple images.
"""
RasterDataSet = Dict[str, Callable[[], DatasetReader]]
"""
Type alias for a dataset containing multiple rasters.
"""

sns.set_theme(
    context="paper", style="whitegrid", palette="husl", rc={"savefig.dpi": 600}
)

pandarallel.initialize()


@enum.unique
class ImageExtents(enum.Enum):
    """
    Tracks the extents of the images in world coordinates.
    """

    IMAGE_ID = "image_id"
    """
    The unique ID of the image.
    """

    X1 = "x1"
    """
    The x-coordinate of the top left corner of the image.
    """
    Y1 = "y1"
    """
    The y-coordinate of the top left corner of the image.
    """
    X2 = "x2"
    """
    The x-coordinate of the bottom right corner of the image.
    """
    Y2 = "y2"
    """
    The y-coordinate of the bottom right corner of the image.
    """


def _split_image_grid(
    image: Image.Image,
) -> Iterable[Tuple[Image.Image, np.array]]:
    """
    Splits an image into a grid of sub-images.

    Args:
        image: The image to split.

    Returns:
        An iterable of sub-images, along with their pixel offsets relative to
        the original image.

    """
    width, height = image.size
    for y in range(0, height, height // 4):
        for x in range(0, width, width // 4):
            yield (
                image.crop((x, y, x + width // 4, y + height // 4)),
                np.array([x, y]),
            )


def detect_flowers(
    images: ImageDataSet,
    *,
    weights_file: Path,
    session_name: str,
    batch_size: int,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Performs flower detection for the images in a session.

    Args:
        images: The images to detect flowers in.
        weights_file: The location of the file to load the model weights from.
        session_name: The name of the session.
        batch_size: The batch size to use for inference.
        dry_run: If true, just return an empty dataframe without actually
            doing detection.

    Returns:
        A dataframe containing the detected flowers.

    """
    # Load the model.
    model = YOLO(weights_file)

    # Split individual images up into more manageable chunks.
    def _iter_patches() -> Iterable[Tuple[str, np.array, np.array]]:
        for image_id_, image in images.items():
            for i, (patch, offsets_) in enumerate(_split_image_grid(image())):
                yield f"{image_id_}_patch_{i}", patch, offsets_

    # Predict on batches.
    detection_columns = [
        DetectionColumns.X1.value,
        DetectionColumns.Y1.value,
        DetectionColumns.X2.value,
        DetectionColumns.Y2.value,
    ]
    results = [
        pd.DataFrame(
            data={},
            columns=detection_columns
            + [
                DetectionColumns.IMAGE_ID.value,
                DetectionColumns.CONFIDENCE.value,
                DetectionColumns.SESSION.value,
            ],
        )
    ]
    if dry_run:
        return results[0]

    for batch in batch_iter(_iter_patches(), batch_size=batch_size):
        # "unzip" the batch.
        image_ids, image_batch, offsets = zip(*batch)
        # Perform inference.
        image_batch = [im.convert("RGB") for im in image_batch]
        yolo_results = model.predict(
            image_batch, imgsz=image_batch[0].size[::-1]
        )

        # Convert the results to a dataframe.
        for image_id, image_results, offset in zip(
            image_ids, yolo_results, offsets
        ):
            results_df = pd.DataFrame(
                data=image_results.boxes.xyxyn.cpu().numpy(),
                columns=detection_columns,
            )
            # Convert to pixels.
            results_df *= np.tile(np.array(image_batch[0].size), (2,))
            # Apply the offsets, which shifts the coordinates into
            # pixel-space for the input image instead of the patches.
            results_df += np.tile(offset, (2,))

            # Add additional columns.
            results_df[
                DetectionColumns.CONFIDENCE.value
            ] = image_results.boxes.conf.cpu().numpy()
            results_df[DetectionColumns.BOX_NUM.value] = np.arange(
                len(results_df)
            )
            results_df[DetectionColumns.IMAGE_ID.value] = image_id

            results.append(results_df)

    all_results = pd.concat(results, ignore_index=True)
    # Add a column for the session name.
    all_results[DetectionColumns.SESSION.value] = session_name
    return all_results


def find_image_extents(
    images: ImageDataSet,
    *,
    camera_config: Dict[str, CameraConfig],
    session_name: str,
    dem_dataset: RasterDataSet,
) -> List[Feature]:
    """
    Finds the extents of each image in the dataset in real-world coordinates.

    Args:
        images: The images to find the shapes of. It assumes they all have
            the same size.
        camera_config: The camera configuration.
        session_name: The name of the session we are finding extents for.
        dem_dataset: The corresponding DEM for this session.

    Returns:
        Shapefile data for the image extents.

    """
    camera_config = camera_config[f"{session_name}_cameras"]
    dem_dataset = dem_dataset[f"{session_name}_dem"]()

    # Find the image size.
    image = next(iter(images.values()))
    width_px, height_px = image().size

    shapes = []
    transformer = CameraTransformer(dem_dataset)
    for image_id in images:
        # Project from the image corners into UTM.
        px_to_utm = partial(
            transformer.pixel_to_utm,
            camera_config=camera_config,
            camera_id=image_id,
        )
        try:
            top_left = px_to_utm(np.array([0.0, 0.0]))
            top_right = px_to_utm(np.array([width_px, 0.0]))
            bottom_right = px_to_utm(
                np.array([width_px, height_px], dtype=np.float32)
            )
            bottom_left = px_to_utm(np.array([0.0, height_px]))
        except MissingImageError:
            logger.warning("No transform data for {}, skipping.", image_id)
            continue

        extent = Polygon(
            [top_left, top_right, bottom_right, bottom_left, top_left]
        )
        if not extent.is_valid:
            logger.warning("Extent for {} is invalid, skipping.", image_id)
            continue
        shapes.append(extent)

    return [
        Feature(
            geometry=Geometry.from_dict(mapping(s)),
            properties=Properties(image_id=i, session=session_name),
        )
        for i, s in zip(images, shapes)
    ]


def flowers_to_geographic(
    detections: pd.DataFrame,
    *,
    camera_config: Dict[str, CameraConfig],
    dem_dataset: RasterDataSet,
    session_name: str,
) -> pd.DataFrame:
    """
    Converts the detected flowers to geographic coordinates.

    Args:
        detections: The detected flowers.
        camera_config: The camera configuration.
        dem_dataset: The corresponding DEM for this session.
        session_name: The name of the session we are loading data from.

    Returns:
        The flowers in geographic coordinates.

    """
    coord_columns = [
        DetectionColumns.IMAGE_ID.value,
        DetectionColumns.X1.value,
        DetectionColumns.Y1.value,
        DetectionColumns.X2.value,
        DetectionColumns.Y2.value,
    ]
    dem_dataset = dem_dataset[f"{session_name}_dem"]()
    transformer = CameraTransformer(dem_dataset)
    camera_config = camera_config[f"{session_name}_cameras"]

    # Convert the coordinates to geographic.
    def _convert(row: pd.Series) -> pd.Series:
        image_id = row[DetectionColumns.IMAGE_ID.value]
        camera_id = image_id.split("_patch_")[0]
        # Get the points to transform.
        box_top_left = row[
            [DetectionColumns.X1.value, DetectionColumns.Y1.value]
        ].to_numpy(dtype=np.float32)
        box_bottom_right = row[
            [DetectionColumns.X2.value, DetectionColumns.Y2.value]
        ].to_numpy(dtype=np.float32)

        try:
            top_x, top_y, _ = transformer.pixel_to_utm(
                box_top_left, camera_config=camera_config, camera_id=camera_id
            )
            bottom_x, bottom_y, _ = transformer.pixel_to_utm(
                box_bottom_right,
                camera_config=camera_config,
                camera_id=camera_id,
            )
        except MissingImageError:
            logger.warning(
                "No transformation for image {}, ignoring detections.",
                image_id,
            )
            return pd.Series(index=coord_columns)

        return pd.Series(
            index=coord_columns,
            data=[image_id, top_x, top_y, bottom_x, bottom_y],
        )

    detection_coords = detections[coord_columns]
    geo_detection_coords = detection_coords.apply(_convert, axis="columns")
    detections[coord_columns] = geo_detection_coords
    # Remove data for images with no transformation.
    detections.dropna(subset=coord_columns, inplace=True)
    return detections


def _load_polygons(features: List[Feature]) -> Iterable[Polygon]:
    """
    Loads polygon features from a shapefile.

    Args:
        features: The raw features from Fiona.

    Returns:
        The polygons.

    """
    for feature in features:
        yield Polygon(feature.geometry.coordinates[0])


def _detections_to_polygons(detections: pd.DataFrame) -> Iterable[Polygon]:
    """
    Converts detections from a dataframe to polygons.

    Args:
        detections: The detections.

    Returns:
        The polygons.

    """
    # Extract raw coordinates as an array of points.
    x1, y1, x2, y2 = [
        c.value
        for c in [
            DetectionColumns.X1,
            DetectionColumns.Y1,
            DetectionColumns.X2,
            DetectionColumns.Y2,
        ]
    ]
    coords = detections[[x1, y1, x2, y1, x2, y2, x1, y2, x1, y1]]
    coords = coords.to_numpy().reshape(-1, 2)

    # Five points for every detection box.
    offsets = np.arange(0, len(coords) + 1, 5)
    # One ring for every box.
    num_rings = np.arange(0, len(coords) // 5 + 1)

    return from_ragged_array(
        GeometryType.POLYGON, coords, (offsets, num_rings)
    )


def _query_intersecting(tree: STRtree, poly: Polygon) -> List[int]:
    """
    Queries the tree for the indices of the polygons that intersect the given
    polygon.

    Args:
        tree: The tree.
        poly: The polygon.

    Returns:
        The indices of the polygons that intersect the given polygon.

    """
    indices = tree.query_nearest(poly)
    nearby_polys = tree.geometries[indices]

    # Filter to only ones that actually intersect and are not identical.
    return [
        i
        for i, p in zip(indices, nearby_polys)
        if not p.equals(poly) and p.intersects(poly)
    ]


def prune_duplicate_detections(
    *, detections: pd.DataFrame, image_extents: List[Feature]
) -> pd.DataFrame:
    """
    Finds detections that are duplicates and removes them.

    Args:
        detections: The complete set of detections, in geographic coordinates.
        image_extents: The extents of each of the input images in the dataset.

    Returns:
        The same detections, with duplicates removed.

    """
    # We initially need to find overlapping images so that we know which
    # regions to focus on.
    image_extent_polys = list(_load_polygons(image_extents))
    detection_polys = list(_detections_to_polygons(detections))
    extent_polys_to_id = {
        p: f.properties["image_id"]
        for p, f in zip(image_extent_polys, image_extents)
    }
    detection_polys_to_id = {
        p: r[DetectionColumns.IMAGE_ID.value]
        for p, (_, r) in zip(detection_polys, detections.iterrows())
    }
    detection_polys_to_row = {
        p: i for p, i in zip(detection_polys, detections.index)
    }
    image_extent_tree = STRtree(image_extent_polys)
    detections_tree = STRtree(list(_detections_to_polygons(detections)))

    prune_rows = set()

    # Removes duplicate detections from the region where two input images
    # overlap.
    def _prune_from_intersection(
        extent1: Polygon, extent2: Polygon
    ) -> Set[int]:
        # Find the intersection between the images.
        intersecting_region = intersection(extent1, extent2)
        # Find all detections in that region.
        detections_indices = _query_intersecting(
            detections_tree, intersecting_region
        )
        detections_in_region = detections_tree.geometries[detections_indices]

        extent1_id = extent_polys_to_id[extent1]
        extent2_id = extent_polys_to_id[extent2]

        # Filter to detections from either of the input images.
        image1_indices = set()
        image2_indices = set()
        for detection in detections_in_region:
            image_id = detection_polys_to_id[detection].split("_patch_")[0]
            index = detection_polys_to_row[detection]
            if image_id == extent1_id:
                image1_indices.add(index)
            elif image_id == extent2_id:
                image2_indices.add(index)

        # Filter detections that are already slated for removal.
        image1_indices = image1_indices - prune_rows
        image2_indices = image2_indices - prune_rows

        # Our current, very naive algorithm is just to prune the detections
        # from the image with fewer IDs.
        if len(image1_indices) > len(image2_indices):
            return image2_indices
        else:
            return image1_indices

    for extent in image_extent_polys:
        # Find all intersecting images.
        intersecting_indices = _query_intersecting(image_extent_tree, extent)
        intersecting = image_extent_tree.geometries[intersecting_indices]
        for other_extent in intersecting:
            # Remove duplicates in the intersecting region.
            prune_rows.update(_prune_from_intersection(extent, other_extent))

    # Remove all the duplicates.
    logger.info("Removing {} duplicate detections.", len(prune_rows))
    return detections.drop(iter(prune_rows))


def flowers_to_shapefile(detections: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Outputs the flower detections as a shapefile.

    Args:
        detections: The flower detections, in geo coordinates.

    Returns:
        The flower detection features.

    """
    features = []
    for (_, row), polygon in zip(
        detections.iterrows(), _detections_to_polygons(detections)
    ):
        features.append(
            {
                "geometry": mapping(polygon),
                "properties": {
                    "confidence": row[DetectionColumns.CONFIDENCE.value],
                    "session": row[DetectionColumns.SESSION.value],
                    "image_id": row[DetectionColumns.IMAGE_ID.value],
                    "box_num": row[DetectionColumns.BOX_NUM.value],
                    "plot_num": row[DetectionColumns.PLOT_NUM.value],
                },
            }
        )
    return features


def _label_plots(
    *,
    plot_boundaries: Iterable[Polygon],
    field_config: FieldConfig,
    sessions: List[str],
) -> Iterable[Tuple[Polygon, int, Set[str]]]:
    """
    Computes the plot number for each plot in a shapefile.

    Args:
        plot_boundaries: The boundaries of the plots.
        field_config: The configuration of the field.
        sessions: The sessions that these boundaries are valid for.

    Yields:
        Each plot boundary polygon, along with the corresponding plot number
        and the sessions that this is valid for.

    """
    # Sort plots north-to-south and then west-to-east.
    plot_boundaries = list(plot_boundaries)
    plot_centers = [p.centroid for p in plot_boundaries]
    boundaries_with_centers = [
        (b, c) for b, c in zip(plot_boundaries, plot_centers)
    ]
    # Sort by x coordinate.
    boundaries_with_centers_sorted_x = sorted(
        boundaries_with_centers, key=lambda p: p[1].x
    )
    # Group by row.
    boundaries_with_centers_by_row = list(
        batch_iter(
            boundaries_with_centers_sorted_x,
            batch_size=field_config.num_plots // field_config.num_rows,
        )
    )
    # Sort by y coordinate within rows.
    boundaries_with_centers_sorted = [
        sorted(row, key=lambda p: p[1].y, reverse=True)
        for row in boundaries_with_centers_by_row
    ]
    boundaries_with_centers_sorted = reduce(
        lambda x, y: x + y, boundaries_with_centers_sorted, []
    )
    boundaries_sorted = [b for b, _ in boundaries_with_centers_sorted]

    # Assign real plot numbers to them.
    sessions = set(sessions)
    for i, boundary in enumerate(boundaries_sorted):
        yield boundary, field_config.get_plot_num_row_major(i), sessions


def _label_plots_gt(
    *,
    sampling_regions: Iterable[Polygon],
    field_config: FieldConfig,
    ground_truth: pd.DataFrame,
    sessions: List[str],
) -> Iterable[Tuple[Polygon, int, Set[str]]]:
    """
    Computes the plot number for each ground-truth sampling region in a
    shapefile.

    Args:
        sampling_regions: The GT sampling regions.
        field_config: The configuration of the field.
        ground_truth: The ground truth data.
        sessions: The sessions that these boundaries are valid for.

    Yields:
        Each sampling region polygon, along with the corresponding plot
        number and the sessions that this is valid for.

    """
    sessions = set(sessions)

    for top_plot, bottom_plot in batch_iter(
        _label_plots(
            plot_boundaries=sampling_regions,
            field_config=field_config,
            sessions=[],
        ),
        batch_size=2,
    ):
        top_plot = top_plot[:2]
        bottom_plot = bottom_plot[:2]
        _, top_plot_num = top_plot
        _, bottom_plot_num = bottom_plot
        if top_plot_num != bottom_plot_num:
            # These are single row plots, so we should yield both.
            yield top_plot + (sessions,)
            yield bottom_plot + (sessions,)
            continue

        # Since we only sample either the top or bottom row, we have to fiter
        # out extraneous ones.
        try:
            plot_rows = ground_truth.loc[[top_plot_num]]
        except KeyError:
            # This plot wasn't measured in the ground-truth data, so we can
            # completely ignore it.
            continue

        alternate_row = plot_rows[GroundTruthColumns.USED_ALTERNATE_ROW.value]
        if alternate_row.any():
            yield bottom_plot + (
                set(
                    plot_rows.loc[
                        alternate_row, GroundTruthColumns.SESSION.value
                    ]
                    .unique()
                    .tolist()
                )
                & sessions,
            )
        if not alternate_row.all():
            yield top_plot + (
                set(
                    plot_rows.loc[
                        ~alternate_row, GroundTruthColumns.SESSION.value
                    ]
                    .unique()
                    .tolist()
                )
                & sessions,
            )


def _shift_plot(plot_boundary: Polygon, shift_amount: float) -> Polygon:
    """
    Shifts the plot boundary by a certain amount, along the longer axis of
    the plot.

    Args:
        plot_boundary: The plot boundary.
        shift_amount: The amount to shift the plot by in the x direction.

    Returns:
        The shifted plot boundary.

    """
    # Find the slope of the sides.
    point1, point2, point3 = plot_boundary.exterior.coords[:3]
    slope1 = (point2[1] - point1[1]) / (point2[0] - point1[0])
    slope2 = (point3[1] - point2[1]) / (point3[0] - point2[0])
    nearest_horizontal_slope = min(slope1, slope2)

    # Compute the amount of shift in each direction.
    shift_y = shift_amount * nearest_horizontal_slope

    return affinity.translate(plot_boundary, xoff=shift_amount, yoff=shift_y)


def _label_plots_gt_monte_carlo(
    labeled_plots: Iterable[Tuple[Polygon, int, Set[str]]],
    *,
    max_shift_amount: float,
) -> Iterable[Tuple[Polygon, int, Set[str]]]:
    """
    Applies random shifts to the  plots in order to simulate the fact that
    Dalton might not always sample the exact center of the plot.

    Args:
        labeled_plots: The previously-labeled GT sampling locations.
        max_shift_amount: The maximum amount to shift in the x direction.

    Returns:
        The augmented labeled plots.

    """
    for boundary, plot_num, sessions in labeled_plots:
        shifted_plot = _shift_plot(
            boundary,
            np.random.uniform(-max_shift_amount, max_shift_amount),
        )
        yield shifted_plot, plot_num, sessions


def _find_detections_in_regions(
    *,
    detections: pd.DataFrame,
    labeled_plots: Iterable[Tuple[Polygon, int, Set[str]]],
) -> pd.DataFrame:
    """
    Finds detections within specific plots, and adds the appropriate plot
    numbers to the detections dataframe.

    Args:
        detections: The detections.
        labeled_plots: The plot polygons with associated plot numbers and
            valid sessions.

    Returns:
        The detections with the plot number column.

    """
    # Detection RTree for fast intersection calculations.
    detection_polys = list(_detections_to_polygons(detections))
    detections_tree = STRtree(detection_polys)
    detection_polys_to_row = {
        p: i for p, i in zip(detection_polys, detections.index)
    }

    # Make sure it has a column for plot numbers.
    if DetectionColumns.PLOT_NUM.value not in detections:
        detections.insert(0, DetectionColumns.PLOT_NUM.value, np.nan)
    # We also use a column to keep track of whether this detection is in a plot.
    detections.insert(0, "in_plot", False)

    # Index by sessions as well.
    detections.set_index(
        DetectionColumns.SESSION.value, append=True, drop=False, inplace=True
    )

    for plot_boundary, plot_num, plot_sessions in labeled_plots:
        # Figure out which detections intersect this plot.
        plot_detection_indices = _query_intersecting(
            detections_tree, plot_boundary
        )
        plot_detections = detections_tree.geometries[plot_detection_indices]

        # Edit the corresponding rows in the dataframe.
        detection_rows = [detection_polys_to_row[p] for p in plot_detections]
        try:
            detections.loc[
                (detection_rows, list(plot_sessions)),
                (DetectionColumns.PLOT_NUM.value, "in_plot"),
            ] = [plot_num, True]
        except KeyError:
            # In this case, none of the rows met the session criteria. No
            # changes to the detections frame are necessary.
            pass

    # Keep only the detections that are in a plot.
    detections = detections[detections["in_plot"]].copy()
    detections.drop(columns=["in_plot"], inplace=True)
    # Remove the temporary index we added.
    return detections.droplevel(DetectionColumns.SESSION.value)


def _find_detections_in_plots(
    *,
    detections: pd.DataFrame,
    plot_boundaries: List[Feature],
    field_config: FieldConfig,
    sessions: List[str],
) -> pd.DataFrame:
    """
    Adds the plot number to the detections `DataFrame`.

    Args:
        detections: The original detections.
        plot_boundaries: The plot boundary shapes.
        field_config: The field configuration.
        sessions: List of sessions that these plot boundaries are valid for.

    Returns:
        The same detections, with a plot number column.

    """
    return _find_detections_in_regions(
        detections=detections,
        labeled_plots=_label_plots(
            plot_boundaries=_load_polygons(plot_boundaries),
            field_config=field_config,
            sessions=sessions,
        ),
    )


def find_detections_in_plots_pre_september(
    *,
    detections: pd.DataFrame,
    plot_boundaries: List[Feature],
    field_config: FieldConfig,
) -> pd.DataFrame:
    """
    Adds the plot number to the detections `DataFrame`, for data from before
    September 1.

    Args:
        detections: The original detections.
        plot_boundaries: The plot boundary shapes.
        field_config: The field configuration.

    Returns:
        The same detections, with a plot number column.

    """
    # Find the sessions from before September.
    all_sessions = detections[DetectionColumns.SESSION.value].unique().tolist()
    valid_sessions = [s for s in all_sessions if s < "2023-09-01"]

    return _find_detections_in_regions(
        detections=detections,
        labeled_plots=_label_plots(
            plot_boundaries=_load_polygons(plot_boundaries),
            field_config=field_config,
            sessions=valid_sessions,
        ),
    )


def find_detections_in_plots_post_september(
    *,
    detections: pd.DataFrame,
    plot_boundaries: List[Feature],
    field_config: FieldConfig,
) -> pd.DataFrame:
    """
    Adds the plot number to the detections `DataFrame`, for data from after
    September 1.

    Args:
        detections: The original detections.
        plot_boundaries: The plot boundary shapes.
        field_config: The field configuration.

    Returns:
        The same detections, with a plot number column.

    """
    # Find the sessions from before September.
    all_sessions = detections[DetectionColumns.SESSION.value].unique().tolist()
    valid_sessions = [s for s in all_sessions if s >= "2023-09-01"]

    return _find_detections_in_regions(
        detections=detections,
        labeled_plots=_label_plots(
            plot_boundaries=_load_polygons(plot_boundaries),
            field_config=field_config,
            sessions=valid_sessions,
        ),
    )


def _find_detections_in_gt_sampling_regions(
    *,
    detections: pd.DataFrame,
    gt_sampling_regions: List[Feature],
    field_config: FieldConfig,
    ground_truth: pd.DataFrame,
    sessions: List[str],
) -> pd.DataFrame:
    """
    Adds the plot number to the detections `DataFrame`, based on which
    detections fall within that plot's ground-truth sampling region.

    Args:
        detections: The original detections.
        gt_sampling_regions: The GT sampling region shapes.
        field_config: The field configuration.
        ground_truth: The ground truth data.
        sessions: List of sessions that these plot boundaries are valid for.

    Returns:
        The same detections, with a plot number column.

    """
    all_detections = []

    for sample_num in range(40):
        sample_detections = _find_detections_in_regions(
            detections=detections.copy(),
            labeled_plots=_label_plots_gt_monte_carlo(
                _label_plots_gt(
                    sampling_regions=_load_polygons(gt_sampling_regions),
                    field_config=field_config,
                    ground_truth=ground_truth,
                    sessions=sessions,
                ),
                max_shift_amount=2.0,
            ),
        )

        sample_detections[DetectionColumns.SAMPLE.value] = sample_num
        all_detections.append(sample_detections)

    return pd.concat(all_detections)


def find_detections_in_gt_sampling_regions_pre_september(
    *,
    detections: pd.DataFrame,
    gt_sampling_regions: List[Feature],
    field_config: FieldConfig,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds the plot number to the detections `DataFrame`, based on which
    detections fall within that plot's ground-truth sampling region, for data
    from before September 1.

    Args:
        detections: The original detections.
        gt_sampling_regions: The GT sampling region shapes.
        field_config: The field configuration.
        ground_truth: The ground truth data.

    Returns:
        The same detections, with a plot number column.

    """
    # Find the sessions from before September.
    all_sessions = detections[DetectionColumns.SESSION.value].unique().tolist()
    valid_sessions = [s for s in all_sessions if s < "2023-09-01"]

    return _find_detections_in_gt_sampling_regions(
        detections=detections,
        gt_sampling_regions=gt_sampling_regions,
        field_config=field_config,
        ground_truth=ground_truth,
        sessions=valid_sessions,
    )


def find_detections_in_gt_sampling_regions_post_september(
    *,
    detections: pd.DataFrame,
    gt_sampling_regions: List[Feature],
    field_config: FieldConfig,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds the plot number to the detections `DataFrame`, based on which
    detections fall within that plot's ground-truth sampling region, for data
    from after September 1.

    Args:
        detections: The original detections.
        gt_sampling_regions: The GT sampling region shapes.
        field_config: The field configuration.
        ground_truth: The ground truth data.

    Returns:
        The same detections, with a plot number column.

    """
    # Find the sessions from before September.
    all_sessions = detections[DetectionColumns.SESSION.value].unique().tolist()
    valid_sessions = [s for s in all_sessions if s >= "2023-09-01"]

    return _find_detections_in_gt_sampling_regions(
        detections=detections,
        gt_sampling_regions=gt_sampling_regions,
        field_config=field_config,
        ground_truth=ground_truth,
        sessions=valid_sessions,
    )


def load_ground_truth(
    *, raw_gt_data: pd.DataFrame, session_name: str
) -> pd.DataFrame:
    """
    Loads the ground truth data from the spreadsheet, and formats it how we
    expect.

    Args:
        raw_gt_data: The raw spreadsheet data.
        session_name: The name of the session.

    Returns:
        The correctly-formatted ground-truth data.

    """
    # Rename the columns that directly carry over.
    raw_gt_data.rename(
        {
            "Plot": GroundTruthColumns.PLOT.value,
            "# of White Flowers": GroundTruthColumns.TRUE_COUNT.value,
        },
        axis="columns",
        inplace=True,
    )

    # Determine which row is used based on the notes.
    raw_gt_data[GroundTruthColumns.USED_ALTERNATE_ROW.value] = raw_gt_data[
        "Notes"
    ].str.contains("right row", case=False)

    # Session is specified manually.
    raw_gt_data[GroundTruthColumns.SESSION.value] = session_name
    # The person collecting was always Dalton.
    raw_gt_data[GroundTruthColumns.PERSON.value] = "DW"

    # Index by plot number.
    raw_gt_data.set_index(
        GroundTruthColumns.PLOT.value,
        inplace=True,
    )
    raw_gt_data.sort_index(inplace=True)

    # Keep only the columns we care about.
    return raw_gt_data[
        [
            GroundTruthColumns.TRUE_COUNT.value,
            GroundTruthColumns.USED_ALTERNATE_ROW.value,
            GroundTruthColumns.SESSION.value,
            GroundTruthColumns.PERSON.value,
        ]
    ]


def add_plot_index(
    plot_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds the correct plot index to data with a plot number column.

    Args:
        plot_data: The raw plot data, with a plot number column.

    Returns:
        The same data, indexed by field plot number.

    """
    plot_data[CountingColumns.PLOT.value] = plot_data[
        DetectionColumns.PLOT_NUM.value
    ].astype("uint64")

    # Set the index.
    plot_data = plot_data.set_index(CountingColumns.PLOT.value)
    plot_data.index.name = CountingColumns.PLOT.value
    plot_data.sort_index(inplace=True)

    return plot_data


def plot_ground_truth_vs_predicted(
    counts_with_gt: pd.DataFrame,
) -> plot.Figure:
    """
    Creates a scatter plot showing the ground-truth vs. predicted counts.

    Args:
        counts_with_gt: The merged counting results and ground-truth data.

    Returns:
        The plot that it created.

    """
    counts_with_gt.reset_index(
        level=GroundTruthColumns.DAP.value, inplace=True
    )

    # Plot the regression.
    axes = sns.catplot(
        data=counts_with_gt,
        x=GroundTruthColumns.TRUE_COUNT.value,
        y=CountingColumns.COUNT.value,
        hue=CountingColumns.DAP.value,
    )
    axes.set_titles("Predicted vs. Ground-Truth Counts")
    axes.set(xlabel="Ground-Truth", ylabel="Automatic")

    return plot.gcf()
