"""
Encapsulates custom callbacks to use.
"""


import abc
from pathlib import Path
from typing import Any, Dict, Optional, Union

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.layers as layers
from loguru import logger

from ..type_helpers import DatasetOutput
from .visualization import visualize_heat_maps


class _ImageLoggingCallback(callbacks.Callback, abc.ABC):
    """
    Superclass for callbacks that log images or data derived from processed
    images.
    """

    def __init__(
        self,
        *,
        log_dir: Union[str, Path],
        model: keras.Model,
        dataset: tf.data.Dataset,
        log_period: int = 1,
        num_images_per_batch: int = 3,
        max_num_batches: Optional[int] = None,
    ):
        """
        Args:
            log_dir: The directory to write output logs to.
            model: The model to run.
            log_period: Specifies that we want to log every this many epochs.
                This can be increased if logging is getting too expensive.
            num_images_per_batch: Maximum number of images to log for each
                batch in the dataset.
            max_num_batches: The maximum number of batches to log data from. If
                not specified, it will log from all of them.
        """
        super().__init__()

        # Create the SummaryWriter to use.
        self.__writer = tf.summary.create_file_writer(Path(log_dir).as_posix())

        self.__model = model
        self.__dataset = dataset
        self.__log_period = log_period
        self.__num_images_per_batch = num_images_per_batch

        if max_num_batches is not None:
            # Limit the dataset to a set number of batches.
            self.__dataset = self.__dataset.take(max_num_batches)

    @property
    def _model(self) -> keras.Model:
        """
        Returns:
            The model that we are using.
        """
        return self.__model

    @property
    def _writer(self) -> tf.summary.SummaryWriter:
        return self.__writer

    def _save_image(self, *args: Any, **kwargs: Any) -> None:
        """
        Saves an image to Tensorboard, using the internal `SummaryWriter`.

        Args:
            *args: Will be forwarded to `tf.summary.image`.
            **kwargs: Will be forwarded to `tf.summary.image`.

        """
        with self.__writer.as_default():
            tf.summary.image(
                *args, max_outputs=self.__num_images_per_batch, **kwargs
            )

    def _truncate_batch(self, batch: tf.Tensor) -> tf.Tensor:
        """
        Shorthand for truncating a batch of data to the first
        `num_images_per_batch` values.

        Args:
            batch: The batch to truncate.

        Returns:
            The truncated batch.

        """
        return batch[: self.__num_images_per_batch]

    @abc.abstractmethod
    def _log_batch(
        self,
        *,
        inputs: DatasetOutput,
        targets: DatasetOutput,
        epoch: int,
        batch_num: int,
    ) -> None:
        """
        Logs data for a single batch.

        Args:
            inputs: The input data that the model was run with.
            targets: The corresponding target data.
            epoch: The epoch number.
            batch_num: The batch number.

        """

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        if epoch % self.__log_period != 0:
            # Skip logging this epoch.
            return
        logger.debug("Logging with {}...", self.__class__.__name__)

        for batch_num, (input_batch, target_batch) in enumerate(
            self.__dataset
        ):
            logger.debug("Logging for batch {}.", batch_num)
            self._log_batch(
                inputs=input_batch,
                targets=target_batch,
                epoch=epoch,
                batch_num=batch_num,
            )


class LogDensityMaps(_ImageLoggingCallback):
    """
    Callback that logs density map visualizations to TensorBoard at the end of
    every epoch.
    """

    def __init__(
        self, *args: Any, max_density_threshold: float = 0.02, **kwargs: Any,
    ):
        """
        Args:
            *args: Will be forwarded to superclass.
            max_density_threshold: Density threshold to use for colorization.
                Any pixel with this density or more will show up as the maximum
                density color.
            **kwargs: Will be forwarded to superclass.
        """
        super().__init__(*args, **kwargs)

        self.__density_threshold = max_density_threshold

    def _log_batch(
        self,
        *,
        inputs: DatasetOutput,
        targets: DatasetOutput,
        epoch: int,
        batch_num: int,
    ) -> None:
        image_batch = inputs["image"]
        true_density_maps = targets["density_map"]
        # Don't bother with images that we're not going to log anyway.
        image_batch = self._truncate_batch(image_batch)
        true_density_maps = self._truncate_batch(true_density_maps)

        # Generate predicted density maps.
        predictions = self.__model.predict_on_batch(inputs)
        density_maps = predictions["density_map"]

        # Generate visualizations.
        predicted_density_vis = visualize_heat_maps(
            images=image_batch,
            features=density_maps,
            max_color_threshold=self.__density_threshold,
        )
        true_density_vis = visualize_heat_maps(
            images=image_batch,
            features=true_density_maps,
            max_color_threshold=self.__density_threshold,
        )

        # Save the visualizations.
        self._save_image(
            f"Density Map (Batch {batch_num})",
            predicted_density_vis,
            step=epoch,
        )
        self._save_image(
            f"GT Density Map (Batch {batch_num})",
            true_density_vis,
            step=epoch,
        )


class LogClassActivations(_ImageLoggingCallback):
    """
    Callback that logs the activation maps for each class at the end of each
    epoch when performing discrete counting.
    """

    def __init__(
        self, *args: Any, num_classes: int, **kwargs: Any,
    ):
        """
        Args:
            *args: Will be forwarded to the superclass.
            num_classes: The number of output classes that we have.
            **kwargs: Will be forwarded to the superclass.
        """
        super().__init__(*args, **kwargs)

        self.__num_classes = num_classes

        # Extract the activations from the model.
        activation_layer = self._model.get_layer("activation_maps_pac")
        activation_output = activation_layer.get_output_at(0)
        # This output will go through softmax after global average pooling
        # anyway, so applying softmax here should give us a reasonable
        # visualization of the activation.
        activation_softmax = layers.Softmax()(activation_output)
        self.__extractor = tf.keras.Model(
            inputs=self._model.inputs, outputs=[activation_softmax]
        )

    def _log_batch(
        self,
        *,
        inputs: DatasetOutput,
        targets: DatasetOutput,
        epoch: int,
        batch_num: int,
    ) -> None:
        image_batch = inputs["image"]
        # Don't bother with data that we're not going to log anyway.
        image_batch = self._truncate_batch(image_batch)

        # Retrieve the activations.
        activations = self.__extractor(image_batch)
        activations = self._truncate_batch(activations)

        # Generate the heatmaps for each class.
        for class_num in range(self.__num_classes):
            class_activations = activations[:, :, :, class_num]
            # visualize_heat_maps expects a 4D input.
            class_activations = tf.expand_dims(class_activations, axis=3)

            class_visualizations = visualize_heat_maps(
                images=image_batch,
                features=class_activations,
                # Softmax or sigmoid activation will put everything between 0
                # and 1.
                max_color_threshold=1.0,
            )

            # Save the heatmaps.
            with self._writer.as_default():
                tf.summary.image(
                    f"Class {class_num} Activations (Batch {batch_num})",
                    class_visualizations,
                    step=epoch,
                )
