"""
Dataset for loading ESRI shapefiles.
"""


from typing import List, Dict, Any
from kedro.io import AbstractVersionedDataSet, Version
import fiona
from pathlib import PurePosixPath, Path
from fiona import crs


class ShapefileDataSet(AbstractVersionedDataSet):
    """
    Dataset for loading ESRI shapefiles.
    """

    def __init__(
        self,
        filepath: str,
        version: Version | None = None,
        schema: Dict[str, Any] = {},
        epsg_crs: int = 4362,
    ):
        """
        Args:
            filepath: The path to the shapefile.
            version: The dataset version.
            schema: The schema of the shapefile.
            epsg_crs: The EPSG code of the CRS of the shapefile.

        """
        super().__init__(filepath=PurePosixPath(filepath), version=version)
        self.__schema = schema
        self.__epsg_crs = epsg_crs

    def _load(self) -> List[fiona.model.Feature]:
        """
        Loads the shapefile data.

        Returns:
            The shapefile contents, as read by `fiona`.

        """
        shape_path = self._get_load_path()
        with fiona.open(shape_path.as_posix()) as shapefile:
            return list(shapefile)

    def _save(self, data: List[Dict[str, Any]]) -> None:
        """
        Saves the shapefile data.

        Args:
            data: The data to save, as a structure writable by `fiona`.

        """
        shape_path = self._get_save_path()
        # Create missing directories.
        Path(shape_path).parent.mkdir(parents=True, exist_ok=True)

        with fiona.open(
            shape_path.as_posix(),
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
        return dict(
            version=self._version,
            schema=self.__schema,
            epsg_crs=self.__epsg_crs,
        )
