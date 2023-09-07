"""
Utilities for converting between camera and geographic coordinates.
"""
from typing import Dict, Tuple
from xml.etree import ElementTree as ET

import numpy as np
from pydantic.dataclasses import dataclass
from pygeodesy import EcefKarney, toUtm8

from src.cotton_counter.type_helpers import ArbitraryTypesConfig
from rasterio import DatasetReader
from loguru import logger


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
    def load(cls, camera_xml: str) -> "CameraConfig":
        """
        Initializes from a camera.xml file (from Metashape), and an
        additional dictionary of parameters loaded by Kedro.

        Args:
            camera_xml: The camera XML data.

        Returns:
            The initialized dataclass.

        """
        # Load the XML.
        root = ET.fromstring(camera_xml)
        chunk_node = root.find("chunk")

        # Load camera intrinsics.
        sensor_node = chunk_node.find("sensors")[0]
        properties = sensor_node.findall("property")
        properties = {p.attrib["name"]: p for p in properties}
        focal_length = float(properties["focal_length"].attrib["value"])
        pixel_width = float(properties["pixel_width"].attrib["value"])
        pixel_height = float(properties["pixel_height"].attrib["value"])

        resolution_node = sensor_node.find("resolution")
        image_width = float(resolution_node.attrib["width"])
        image_height = float(resolution_node.attrib["height"])
        f_x = focal_length / pixel_width
        f_y = focal_length / pixel_height
        # Metashape distortion removal should ensure that the camera is
        # centered.
        camera_intrinsics = np.array(
            [
                [f_x, 0, image_width / 2],
                [0, f_y, image_height / 2],
                [0, 0, 1],
            ]
        )

        # Load the chunk transform.
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


class DemHeightEstimator:
    """
    Helper class for estimating DEM heights in a robust way.
    """

    def __init__(self, dem_dataset: DatasetReader):
        """
        Args:
            dem_dataset: The DEM to read height data from.

        """
        self.__dem_dataset = dem_dataset

        # Keeps track of the previous height measurement.
        self.__previous_height = self.__dem_dataset.statistics(1).min

    def __get_dem_height(self, point_xy: np.array) -> float:
        """
        Computes the height at a particular point on the DEM.

        Args:
            point_xy: The point to compute the height at, in the form
                [x, y].

        Returns:
            The height at the point.

        Raises:
            IndexError: If the point is out of bounds.

        """
        height = next(
            self.__dem_dataset.sample([point_xy.tolist()], indexes=1)
        )[0]
        if height in self.__dem_dataset.nodatavals:
            logger.warning("Pont {} is out of bounds.".format(point_xy))
            raise IndexError(f"Point {point_xy} is out of bounds.")
        return height

    def dist_to_dem(self, point: np.array) -> float:
        """
        Finds the height of the surface at this location. If the location is
        out of bounds, it will use the height of the previous location,
        which is assumed to be close.

        Args:
            point: The point to compute the height at, in the form
                [x, y, z].

        Returns:
            The height at the point.

        """
        try:
            surface_height = self.__get_dem_height(point[:2])
            self.__previous_height = surface_height
        except IndexError:
            # It's out of bounds. We'll just use the previous height.
            surface_height = self.__previous_height

        return point[-1] - surface_height


class CameraTransformer:
    """
    Transforms points from camera to geographic coordinates.
    """

    def __init__(self, dem_dataset: DatasetReader):
        """
        Args:
            dem_dataset: The associated DEM, so we can use depth information.

        """
        self.__dem_dataset = dem_dataset

    def __find_dem_intersection(
        self,
        point1: np.array,
        point2: np.array,
        max_iters: int = 30,
        eps: float = 0.01,
    ) -> np.array:
        """
        Finds the approximate point at which a line crosses the DEM.

        Args:
            point1: The first point defining the line.
            point2: The second point defining the line.
            max_iters: The maximum number of refinement iterations to
                perform. This sets an upper bound on computation in cases
                where the DEM is somewhat sparse.
            eps: If the distance between the DEM and a point is below this
                value, it will return this point.

        Returns:
            The closest point on the DEM to the line, in the form
            [x, y, z].

        """
        step = 1.0
        point = point1
        slope = point2 - point1
        crossed_once = False
        num_iters = 0
        estimator = DemHeightEstimator(self.__dem_dataset)
        dist_to_dem = estimator.dist_to_dem(point)

        while np.abs(dist_to_dem) > eps and num_iters < max_iters:
            point += slope * step
            new_dist_to_dem = estimator.dist_to_dem(point)

            if new_dist_to_dem * dist_to_dem < 0:
                # We crossed the surface. Start going the other way in smaller
                # steps.
                step /= -2.0
                crossed_once = True
            elif np.abs(new_dist_to_dem) > np.abs(dist_to_dem):
                # We overshot, and now we're getting farther away. Go the other
                # direction.
                step *= -2.0
            elif not crossed_once:
                # We're going the right direction, but haven't crossed yet.
                # Increase our step size.
                step *= 2.0

            dist_to_dem = new_dist_to_dem
            num_iters += 1

        return point

    @staticmethod
    def __camera_to_utm(
        point: np.array, *, camera_config: CameraConfig, camera_id: str
    ) -> np.array:
        """
        Converts camera coordinates to UTM.

        Args:
            point: The point to convert in camera coordinates, as a 3-element
                array of `[x, y, s]`.
            camera_config: The camera configuration.
            camera_id: The ID of the camera we are transforming from.

        Returns:
            The easting, northing, and height of the pixel in UTM.

        """
        camera_transform = camera_config.camera_transforms[camera_id]
        chunk_transform = camera_config.chunk_transform

        camera_ecef = (
            chunk_transform @ camera_transform @ np.append(point, [1.0])
        )

        ecef = EcefKarney()
        camera_latlon = ecef.reverse(*camera_ecef[:3])
        camera_utm = toUtm8(camera_latlon.lat, camera_latlon.lon)
        return np.array(
            [camera_utm.easting, camera_utm.northing, camera_latlon.height]
        )

    def pixel_to_utm(
        self, pixel: np.array, *, camera_config: CameraConfig, camera_id: str
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
        intrinsics = camera_config.camera_intrinsics
        intrinsics_inv = np.linalg.inv(intrinsics)

        # Convert pixels to camera coordinates.
        cam_coords = intrinsics_inv @ np.append(pixel, [1.0])

        # Find the line in 3D space that this pixel falls on.
        cam_utm_point_1 = self.__camera_to_utm(
            np.zeros(3), camera_config=camera_config, camera_id=camera_id
        )
        cam_utm_point_2 = self.__camera_to_utm(
            cam_coords,
            camera_config=camera_config,
            camera_id=camera_id,
        )

        # Now, find the intersection with the DEM.
        return tuple(
            self.__find_dem_intersection(
                cam_utm_point_1, cam_utm_point_2
            ).tolist()
        )
