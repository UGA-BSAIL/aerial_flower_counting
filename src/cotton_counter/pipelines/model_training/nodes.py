"""
Defines nodes for the `model_training` pipeline.
"""


import cv2
import numpy as np
import tensorflow as tf

from .model.load_from_dataset import extract_model_input


def train_model(
    training_data: tf.data.Dataset,
    testing_data: tf.data.Dataset,
    *,
    map_height: int,
    map_width: int,
    sigma: int,
    batch_size: int,
    num_prefetch_batches: int
) -> None:
    """
    Trains the model.

    Args:
        training_data: The training dataset to use.
        testing_data: The testing dataset to use.
        map_height: The height of the density maps to create, in px.
        map_width: The width of the density maps to created, ix px.
        sigma:
            The standard deviation in pixels to use for the applied gaussian
            filter.
        batch_size: The size of the batches that we generate.
        num_prefetch_batches: The number of batches to prefetch into memory.
            Increasing this can increase performance at the expense of memory
            usage.

    """
    extraction_kwargs = dict(
        map_shape=(map_height, map_width),
        sigma=sigma,
        batch_size=batch_size,
        num_prefetch_batches=num_prefetch_batches,
    )
    training_input = extract_model_input(training_data, **extraction_kwargs)

    model_input = next(iter(training_input))
    for image, density_map in zip(
        model_input.images, model_input.density_maps
    ):
        image = image.numpy()
        image = cv2.resize(image, (1152, 864))
        cv2.imshow("input", image)

        density_image = density_map.numpy()
        density_image = density_image * (255.0 / np.max(density_image))
        density_image = density_image.astype(np.uint8)
        cv2.imshow("density", density_image)

        cv2.waitKey()
