"""
A custom `DataSet` for creating TFRecords files.
"""


from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Optional

import tensorflow as tf
from kedro.io import AbstractVersionedDataSet, Version
from loguru import logger


class TfRecordsDataSet(AbstractVersionedDataSet):
    """
    A custom `DataSet` for creating TFRecords files.
    """

    def __init__(self, filepath: PurePosixPath, version: Optional[Version]):
        """
        Args:
            filepath: The path to the output TFRecords file.
            version: The version information for the `DataSet`.
        """
        super().__init__(PurePosixPath(filepath), version)

    def _load(self) -> tf.data.TFRecordDataset:
        """
        Loads the TFRecord file.

        Returns:
            The raw `tf.data.TFRecordDataset` that it loaded.

        """
        load_path = self._get_load_path()
        logger.debug("Loading TFRecords from {}.", load_path)
        raw_dataset = tf.data.TFRecordDataset([load_path.as_posix()])

        return raw_dataset

    def _save(self, examples: Iterable[tf.train.Example]) -> None:
        """
        Saves data to a TFRecord file.

        Args:
            examples: Examples to write will be pulled from this iterable and
                saved in-order.

        """
        save_path = self._get_save_path()
        logger.debug("Saving TFRecords to {}.", save_path)

        # Make sure all the intermediate directories are created.
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        with tf.io.TFRecordWriter(save_path.as_posix()) as writer:
            for example in examples:
                serialized = example.SerializeToString()
                writer.write(serialized)

        logger.debug("Done saving data.")

    def _exists(self) -> bool:
        path = self._get_load_path()
        return Path(path.as_posix()).exists()

    def _describe(self) -> Dict[str, Any]:
        return dict(version=self._version)
