"""
Dataset for loading large raster files.
"""


from pathlib import Path
from typing import Type, Dict, Any
import rasterio
from pydantic.dataclasses import dataclass
import numpy as np

from kedro.io import AbstractDataSet

from ..type_helpers import ArbitraryTypesConfig


@dataclass(frozen=True, config=ArbitraryTypesConfig)
class RasterConfig:
    """
    Configuration that we use when writing a raster.

    Attributes:
        data: The data that we want to save to the raster.

        crs: The CRS to use for georeferencing.
        transform: The affine transformation to use to map pixel space to
            geographic space.

        driver: The driver to use for saving. Defaults to GeoTiff.
        height: The height of the raster. Defaults to the height of the data
            we are saving.
        width: The width of the raster. Defaults to the width of the data
            we are saving.
        count: Number of dataset bands. Defaults to the size of the third
            dimension of the data we are saving.
        dtype: The raster data type. Defaults to the type of the data we are
            saving.

    """

    data: np.ndarray

    crs: str
    transform: rasterio.transform.AffineTransformer

    driver: str = "GTiff"
    height: int | None = None
    width: int | None = None
    count: int | None = None
    dtype: str | Type | None = None


class RasterDataSet(AbstractDataSet):
    """
    Dataset for loading large raster files.
    """

    def __init__(self, filepath: str):
        """
        Args:
            filepath: The path to the raster file.

        """
        self.__raster_path = Path(filepath)

    def _load(self) -> rasterio.DatasetReader:
        """
        Loads the data from the raster file.

        Returns:
            The data as a `rasterio` dataset.

        """
        return rasterio.open(self.__raster_path)

    def _save(self, raster: RasterConfig) -> None:
        """
        Saves data to the raster file.

        Args:
            raster: The data to write.

        """
        data = raster.data
        if len(data.shape) == 2:
            # Add the third dimension.
            data = np.expand_dims(data, axis=2)
        data_rows, data_cols, data_bands = data.shape

        # Set default values for parameters if necessary.
        width = raster.width if raster.width is not None else data_cols
        height = raster.height if raster.height is not None else data_rows
        count = raster.count if raster.count is not None else data_bands
        dtype = raster.dtype if raster.dtype is not None else data.dtype

        with rasterio.open(
            self.__raster_path,
            "w",
            driver=raster.driver,
            width=width,
            height=height,
            count=count,
            crs=raster.crs,
            transform=raster.transform,
            dtype=dtype,
        ) as dataset:
            # Write each band.
            for i in range(data_bands):
                # Bands are one-indexed.
                dataset.write(data[:, :, i], i + 1)

    def _describe(self) -> Dict[str, Any]:
        """
        Returns:
            Dict that describes the attributes of the dataset.

        """
        return dict(filepath=self.__raster_path.as_posix())
