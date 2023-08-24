"""
Contains nodes for the updated plot counting pipeline.
"""


from typing import Tuple, Iterable, Dict, Callable, Any, List
from functools import partial
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from ..common import batch_iter, DetectionColumns
from pydantic.dataclasses import dataclass
import xml.etree.ElementTree as ET
import cv2
from pygeodesy.ecef import EcefKarney
from pygeodesy.utm import toUtm8
from pandarallel import pandarallel
from ...type_helpers import ArbitraryTypesConfig
from shapely import Polygon
from shapely.geometry import mapping
import enum


ImageDataSet = Dict[str, Callable[[], Image.Image]]
"""
Type alias for a dataset containing multiple images.
"""


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


@dataclass(config=ArbitraryTypesConfig)
class CameraConfig:
    """
    Represents the configuration for the Metashape cameras.

    Attributes:
        chunk_transform: 3D transformation matrix from chunk coordinates to
            ECEF.
        camera_intrinsics: The camera intrinsics matrix.
        camera_transforms: Maps camera IDs to 4D transformation matrices from
            camera coordinates to chunk coordinates.
    """

    chunk_transform: np.ndarray
    camera_intrinsics: np.ndarray
    camera_transforms: Dict[str, np.ndarray]

    @classmethod
    def _load_camera_transforms(
        cls, cameras: ET.Element
    ) -> Dict[str, np.ndarray]:
        """
        Loads the camera transforms from an XML tree.

        Args:
            cameras: The `<cameras>` element from the camera.xml file.

        Returns:
            A mapping of the camera labels to the camera transform matrices.

        """
        return {
            camera.attrib["label"]: np.reshape(
                np.fromstring(camera.find("transform").text, sep=" "), (4, 4)
            )
            for camera in cameras
            if camera.attrib.get("enabled", "true") != "false"
        }

    @classmethod
    def load(
        cls, camera_xml: str, parameters: Dict[str, Any]
    ) -> "CameraConfig":
        """
        Initializes from a camera.xml file (from Metashape), and an
        additional dictionary of parameters loaded by Kedro.

        Args:
            camera_xml: The camera XML data.
            parameters: Additional parameter values.

        Returns:
            The initialized dataclass.

        """
        # Load camera intrinsics.
        f_x = parameters["f_x"]
        f_y = parameters["f_y"]
        c_x = parameters["c_x"]
        c_y = parameters["c_y"]
        camera_intrinsics = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])

        # Load the XML.
        root = ET.fromstring(camera_xml)
        # Load the chunk transform.
        chunk_node = root.find("chunk")
        chunk_transform_node = chunk_node.find("components")[0].find(
            "transform"
        )
        chunk_rotation = np.fromstring(
            chunk_transform_node.find("rotation").text, sep=" "
        )
        chunk_translation = np.fromstring(
            chunk_transform_node.find("translation").text, sep=" "
        )
        chunk_scale = float(chunk_transform_node.find("scale").text)

        chunk_transform = np.eye(4, dtype=np.float32)
        chunk_transform[:3, :3] = (
            np.reshape(chunk_rotation, (3, 3)) * chunk_scale
        )
        chunk_transform[:3, 3] = chunk_translation

        # Load the camera transforms.
        cameras_node = chunk_node.find("cameras")
        camera_transforms = cls._load_camera_transforms(cameras_node)

        return cls(
            chunk_transform=chunk_transform,
            camera_intrinsics=camera_intrinsics,
            camera_transforms=camera_transforms,
        )


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
) -> pd.DataFrame:
    """
    Performs flower detection for the images in a session.

    Args:
        images: The images to detect flowers in.
        weights_file: The location of the file to load the model weights from.
        session_name: The name of the session.
        batch_size: The batch size to use for inference.

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
    results = []
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
                columns=[
                    DetectionColumns.X1.value,
                    DetectionColumns.Y1.value,
                    DetectionColumns.X2.value,
                    DetectionColumns.Y2.value,
                ],
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
    images: ImageDataSet, *, camera_config: CameraConfig
) -> List[Dict[str, Any]]:
    """
    Finds the extents of each image in the dataset in real-world coordinates.

    Args:
        images: The images to find the shapes of. It assumes they all have
            the same size.
        camera_config: The camera configuration.

    Returns:
        Shapefile data for the image extents.

    """
    # Find the image size.
    image = next(iter(images.values()))
    width_px, height_px = image().size

    shapes = []
    for image_id in images:
        # Project from the image corners into UTM.
        px_to_utm = partial(
            _pixel_to_utm, camera_config=camera_config, camera_id=image_id
        )
        top_left = px_to_utm(np.array([0.0, 0.0]))
        top_right = px_to_utm(np.array([width_px, 0.0]))
        bottom_right = px_to_utm(
            np.array([width_px, height_px], dtype=np.float32)
        )
        bottom_left = px_to_utm(np.array([0.0, height_px]))

        shapes.append(
            Polygon([top_left, top_right, bottom_right, bottom_left, top_left])
        )

    return [
        dict(geometry=mapping(s), properties=dict(image_id=i))
        for i, s in zip(images, shapes)
    ]


def _pixel_to_utm(
    pixel: np.array, *, camera_config: CameraConfig, camera_id: str
) -> Tuple[float, float, float]:
    """
    Converts pixels in image-space to UTM.

    Args:
        pixel: The pixel to convert, as a 2-element array of `[x, y]`.
        camera_config: The camera configuration.
        camera_id: The ID of the camera we are transforming from.

    Returns:
        The easting, northing, and height of the pixel in UTM.

    """
    camera_transform = camera_config.camera_transforms[camera_id]
    chunk_transform = camera_config.chunk_transform
    intrinsics = camera_config.camera_intrinsics

    # Convert pixels to camera coordinates.
    cam_coords = cv2.undistortPoints(
        np.expand_dims(pixel, 0), intrinsics, None
    ).squeeze()
    camera_ecef = (
        chunk_transform @ camera_transform @ np.append(cam_coords, [0.0, 1.0])
    )

    ecef = EcefKarney()
    camera_latlon = ecef.reverse(*camera_ecef)
    camera_utm = toUtm8(camera_latlon.lat, camera_latlon.lon)
    return camera_utm.easting, camera_utm.northing, camera_latlon.height


def flowers_to_geographic(
    detections: pd.DataFrame, *, camera_config: CameraConfig
) -> pd.DataFrame:
    """
    Converts the detected flowers to geographic coordinates.

    Args:
        detections: The detected flowers.
        camera_config: The camera configuration.

    Returns:
        The flowers in geographic coordinates.

    """
    coord_columns = [
        DetectionColumns.X1.value,
        DetectionColumns.Y1.value,
        DetectionColumns.X2.value,
        DetectionColumns.Y2.value,
    ]

    # Convert the coordinates to geographic.
    def _convert(row: pd.Series) -> pd.Series:
        camera_id = row[DetectionColumns.IMAGE_ID.value]
        camera_id = camera_id.split("_patch_")[0]
        # Get the points to transform.
        box_top_left = row[
            [DetectionColumns.X1.value, DetectionColumns.Y1.value]
        ].to_numpy(dtype=np.float32)
        box_bottom_right = row[
            [DetectionColumns.X2.value, DetectionColumns.Y2.value]
        ].to_numpy(dtype=np.float32)

        top_x, top_y, _ = _pixel_to_utm(
            box_top_left, camera_config=camera_config, camera_id=camera_id
        )
        bottom_x, bottom_y, _ = _pixel_to_utm(
            box_bottom_right, camera_config=camera_config, camera_id=camera_id
        )

        return pd.Series(
            index=coord_columns,
            data=[top_x, top_y, bottom_x, bottom_y],
        )

    detection_coords = detections[
        [DetectionColumns.IMAGE_ID.value] + coord_columns
    ]
    geo_detection_coords = detection_coords.parallel_apply(
        _convert, axis="columns"
    )
    detections[coord_columns] = geo_detection_coords
    return detections


def flowers_to_shapefile(detections: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Outputs the flower detections as a shapefile.

    Args:
        detections: The flower detections, in geo coordinates.

    Returns:
        The flower detection features.

    """
    features = []
    for _, row in detections.iterrows():
        features.append(
            {
                "geometry": mapping(
                    Polygon(
                        [
                            (
                                row[DetectionColumns.X1.value],
                                row[DetectionColumns.Y1.value],
                            ),
                            (
                                row[DetectionColumns.X2.value],
                                row[DetectionColumns.Y1.value],
                            ),
                            (
                                row[DetectionColumns.X2.value],
                                row[DetectionColumns.Y2.value],
                            ),
                            (
                                row[DetectionColumns.X1.value],
                                row[DetectionColumns.Y2.value],
                            ),
                            (
                                row[DetectionColumns.X1.value],
                                row[DetectionColumns.Y1.value],
                            ),
                        ]
                    )
                ),
                "properties": {
                    "confidence": row[DetectionColumns.CONFIDENCE.value],
                    "session": row[DetectionColumns.SESSION.value],
                    "image_id": row[DetectionColumns.IMAGE_ID.value],
                    "box_num": row[DetectionColumns.BOX_NUM.value],
                },
            }
        )
    return features
