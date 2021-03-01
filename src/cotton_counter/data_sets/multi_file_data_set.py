"""
This is actually a sort of meta-dataset that is designed to wrap other
datasets which produce a file. It take an arbitrary number of the outputs
from one of these datasets and save them to (or load them from) a single
directory.
"""


from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Optional

from kedro.io import AbstractDataSet, AbstractVersionedDataSet, Version
from loguru import logger


class MultiFileDataSet(AbstractVersionedDataSet):
    """
    This is actually a sort of meta-dataset that is designed to wrap other
    datasets which produce a file. It take an arbitrary number of the outputs
    from one of these datasets and save them to (or load them from) a single
    directory.
    """

    def __init__(
        self,
        filepath: PurePosixPath,
        version: Optional[Version],
        dataset: str,
        extension: str,
        **kwargs: Any,
    ):
        """
        Args:
            filepath: The path to the output directory.
            version: The version information for the `DataSet`.
            dataset: The name of the dataset module that we are wrapping.
            extension: Extension to use for created files.
            **kwargs: Will be forwarded to the internal datasets.
        """
        super().__init__(PurePosixPath(filepath), version)

        self.__dataset_type = dataset
        self.__version = version
        self.__dataset_config = kwargs
        self.__extension = extension

        # The internal datasets that we use for saving and loading.
        self.__datasets = []

    def __create_dataset(
        self, index: int, *, base_path: Path
    ) -> AbstractDataSet:
        """
        Creates a new dataset.

        Args:
            index: The index to use in the dataset's name.
            base_path: The location of the directory where the data in this
                dataset is stored.

        Returns:
            The dataset that it created.

        """
        # Construct the file path.
        filepath = base_path / f"data_{index}{self.__extension}"
        logger.debug(
            "Creating {} to save data at {}.", self.__dataset_type, filepath
        )
        config = dict(
            filepath=filepath.as_posix(),
            type=self.__dataset_type,
            **self.__dataset_config,
        )

        return AbstractVersionedDataSet.from_config(
            f"{self.__dataset_type}_{index}",
            config,
            load_version=self.__version.load,
            save_version=self.__version.save,
        )

    def __iter_datasets(
        self, base_path: Optional[Path] = None
    ) -> Iterable[AbstractDataSet]:
        """
        Iterates through all of the internal datasets in-order, creating
        those that don't exist on-the-fly.

        Args:
            base_path: The base path to use when creating datasets. Defaults
                to the save path.

        Yields:
              The internal datasets.

        """
        yield from self.__datasets

        if base_path is None:
            base_path = self._get_save_path()

        # At this point, we need to start adding new ones.
        while True:
            dataset = self.__create_dataset(
                len(self.__datasets), base_path=base_path
            )
            self.__datasets.append(dataset)

            yield dataset

    def _load(self) -> Iterable[Any]:
        # Figure out how many saved files there are.
        load_path = Path(self._get_load_path())
        saved_files = load_path.iterdir()

        for _, dataset in zip(
            saved_files, self.__iter_datasets(base_path=load_path)
        ):
            yield dataset.load()

    def _save(self, data: Iterable[Any]) -> None:
        for data_item, dataset in zip(data, self.__iter_datasets()):
            # Save the data.
            dataset.save(data_item)

    def _exists(self) -> bool:
        path = self._get_load_path()
        return Path(path).exists()

    def _describe(self) -> Dict[str, Any]:
        return dict(
            dataset=self.__dataset_type,
            version=self.__version,
            extension=self.__extension,
            config=self.__dataset_config,
        )
