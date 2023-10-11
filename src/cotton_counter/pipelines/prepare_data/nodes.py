import numpy as np
from typing import Dict, Callable, List, Iterable
from PIL import Image
from functools import partial
from fiona import Feature
from rasterio import DatasetReader
from rasterio.io import MemoryFile
from shapely.geometry import Polygon


ImageDataSet = Dict[str, Callable[[], Image.Image]]
"""
Type alias for a dataset containing multiple images.
"""


def _excess_green(frame: np.array) -> np.array:
    """
    Quantifies the excess green for a particular frame.

    Args:
        frame: The frame to compute excess green for.

    Returns:
        The excess green values for each pixel.

    """
    frame = frame.astype(np.float32) / 255.0
    red = frame[:, :, 0]
    green = frame[:, :, 1]
    blue = frame[:, :, 2]

    excess_green = 2 * green - blue - red

    return ((excess_green + 2) * (255 / 4)).astype(np.uint8)


def _excess_green_pil(image_: Callable[[], Image.Image]) -> Image.Image:
    """
    Same as `_excess_green`, but accepts and returns PIL images.

    Args:
        image_: The image to process.

    Returns:
        The pixel-wise excess green data.

    """
    return Image.fromarray(_excess_green(np.array(image_())), mode="L")


def compute_excess_green(dataset: ImageDataSet) -> ImageDataSet:
    """
    Computes the excess green values for a dataset of images.

    Args:
        dataset: The dataset to compute excess green for.

    Returns:
        The dataset with excess green values computed.

    """

    partitions = {}
    for key, image in dataset.items():
        partitions[key] = partial(_excess_green_pil, image)

    return partitions


def _georeference_images(
    images: ImageDataSet, extents: List[Feature]
) -> Iterable[DatasetReader]:
    """
    Creates georeferenced versions of each image.

    Args:
        images: The images to georeference.
        extents: The pre-computed image extents.

    Yields:
        Each image, as a georeferenced raster dataset.

    """
    # Organize the extents by image.
    id_to_polygon = {}
    for feature in extents:
        key = f"{feature.properties['session']}_{feature.properties['image_id']}"
        id_to_polygon[key] = Polygon(
            feature.geometry.coordinates[0]
        )

    for key, image in images:
        extent = id_to_polygon[key]
        image = image()
        with MemoryFile(image.tobytes()) as memfile:
            with memfile.open() as dataset:


