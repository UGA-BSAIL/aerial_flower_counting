"""
Encapsulates custom callbacks to use.
"""


from pathlib import Path
from typing import Dict, Optional, Union

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.callbacks as callbacks
from loguru import logger

from .density_maps import visualize_density_maps


class LogDensityMaps(callbacks.Callback):
    """
    Callback that logs density map visualizations to TensorBoard at the end of
    every epoch.
    """

    def __init__(
        self,
        *,
        log_dir: Union[str, Path],
        model: keras.Model,
        dataset: tf.data.Dataset,
        log_period: int = 1,
        max_density_threshold: float = 0.02,
    ):
        """
        Args:
            log_dir: The directory to write output logs to.
            model: The model to run.
            log_period: Specifies that we want to log every this many epochs.
                This can be increased if logging is getting too expensive.
            max_density_threshold: Density threshold to use for colorization.
                Any pixel with this density or more will show up as the maximum
                density color.
        """
        super().__init__()

        # Create the FileWriter to use.
        self.__writer = tf.summary.create_file_writer(Path(log_dir).as_posix())

        self.__model = model
        self.__dataset = dataset
        self.__log_period = log_period
        self.__density_threshold = max_density_threshold

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        if epoch % self.__log_period != 0:
            # Skip logging this epoch.
            return
        logger.debug("Logging density map visualizations...")

        for batch_num, (image_batch, _) in enumerate(self.__dataset):
            # Generate predicted density maps.
            image_batch = tf.cast(image_batch["image"], tf.float32)
            predictions = self.__model.predict_on_batch(image_batch)
            density_maps = predictions["density_map"]

            # Generate visualizations.
            density_visualizations = visualize_density_maps(
                images=image_batch,
                density_maps=density_maps,
                max_density_threshold=self.__density_threshold,
            )

            # Save the visualizations.
            with self.__writer.as_default():
                tf.summary.image(
                    f"Density Map (Batch {batch_num})",
                    density_visualizations,
                    step=epoch,
                )
