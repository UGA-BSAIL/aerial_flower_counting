"""
Dataset for loading ESRI shapefiles.
"""


from typing import List, Dict, Any, Iterable
from kedro.io import AbstractDataSet
import fiona
from pathlib import Path
from fiona import crs


class ShapefileDataSet(AbstractDataSet):
    """
    Dataset for loading ESRI shapefiles.
    """

    def __init__(
        self, filepath: str, schema: Dict[str, Any] = {}, epsg_crs: int = 4362
    ):
        """
        Args:
            filepath: The path to the shapefile.
            schema: The schema of the shapefile.
            epsg_crs: The EPSG code of the CRS of the shapefile.

        """
        self.__shape_path = Path(filepath)
        self.__schema = schema
        self.__epsg_crs = epsg_crs

    def _load(self) -> List[Dict[str, Any]]:
        """
        Loads the shapefile data.

        Returns:
            The shapefile contents, as read by `fiona`.

        """
        with fiona.open(self.__shape_path) as shapefile:
            return list(shapefile)

    def _save(self, data: List[Dict[str, Any]]) -> None:
        """
        Saves the shapefile data.

        Args:
            data: The data to save, as a structure writable by `fiona`.

        """
        # Create missing directories.
        self.__shape_path.parent.mkdir(parents=True, exist_ok=True)

        with fiona.open(
            self.__shape_path,
            "w",
            driver="ESRI Shapefile",
            schema=self.__schema,
            crs=crs.from_epsg(self.__epsg_crs),
        ) as shapefile:
            for row in data:
                shapefile.write(row)

    def _describe(self) -> Dict[str, Any]:
        """
        Returns:
            A human-readable description of the dataset.

        """
        return dict(filepath=self.__shape_path.as_posix())
