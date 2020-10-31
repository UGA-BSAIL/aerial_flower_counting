"""
Helpers for performing inference.
"""


from typing import Dict, Optional

import numpy as np
import tensorflow as tf
from loguru import logger

from src.cotton_counter.type_helpers import Vector2I

from .patches import extract_standard_patches


def _make_patch_indices(
    *, image_shape: Vector2I, patch_scale: float, patch_stride: float
) -> np.ndarray:
    """
    Calculates the indices in the flattened input image that correspond to
    each extracted patch for a particular patch scale and stride.

    Args:
        image_shape: The size of the input image, in the form (height, width).
        patch_scale: The patch scale to use.
        patch_stride: The patch stride to use.

    Returns:
        An array, where the rows correspond to the indices in each patch.

    """
    # Create the 2D index array.
    image_size = np.prod(image_shape)
    index_array = np.arange(0, image_size)
    index_array = index_array.reshape(image_shape)

    # Extract the same patches from the index array.
    index_array_4d = np.expand_dims(index_array, axis=0)
    index_array_4d = np.expand_dims(index_array_4d, axis=3)
    index_array_4d = tf.constant(index_array_4d, dtype=tf.int32)
    index_patches = extract_standard_patches(
        index_array_4d, patch_scale=patch_scale, patch_stride=patch_stride
    )

    # Flatten them into index arrays.
    index_patches = index_patches.numpy()
    num_patches = len(index_patches)
    return index_patches.reshape((num_patches, -1))


def _make_patch_overlap_map(
    *, patch_indices: np.ndarray, image_shape: Vector2I,
) -> np.ndarray:
    """
    Creates an array where each value corresponds to the number of patches
    that overlap at that location in the input image.

    Args:
        patch_indices: An array of the patch indices, as produced by
            `_make_patch_indices()`.
        image_shape: The size of the input image, in the form (height, width).

    Returns:
        A 2D integer array of the same shape as `image_shape`.

    """
    image_size = np.prod(image_shape)
    patch_overlaps = np.zeros((image_size,), dtype=np.int32)

    for indices in patch_indices:
        patch_overlaps[indices] += 1

    return patch_overlaps.reshape(image_shape)


def _make_density_map(
    *,
    patch_predictions: np.ndarray,
    patch_indices: np.ndarray,
    patch_overlap_map: np.ndarray,
) -> np.ndarray:
    """
    Creates an approximate density map when given the count predictions for
    each patch and the corresponding input indices for each patch.

    Args:
        patch_predictions: A vector of count predictions for each patch in
            the image.
        patch_indices: An array of the corresponding patch indices,
            as produced by `_make_patch_indices()`.
        patch_overlap_map: The corresponding patch overlap map, as produced by
            `_make_patch_overlap_map()`.

    Returns:
        The 3D pseudo-density-map that it created.

    """
    # Total counts for each pixel.
    counts_per_pixel = np.zeros(patch_overlap_map.size, dtype=np.float32)

    for prediction, indices in zip(patch_predictions, patch_indices):
        counts_per_pixel[indices] += prediction

    # Compute the density map.
    counts_per_pixel = counts_per_pixel.reshape(patch_overlap_map.shape)
    density_map = counts_per_pixel / patch_overlap_map

    return np.expand_dims(density_map, axis=2)


def _classes_to_counts(predictions: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Converts raw predictions from the model to count estimates.

    Args:
        predictions: The raw predictions dictionary from the model.

    Returns:
        The estimated counts.

    """
    discrete_y_hat = predictions["discrete_count"]
    # Convert from the softmax distribution to actual classes.
    discrete_classes = np.argmax(discrete_y_hat, axis=1)

    # In this case, we assume that class zero means there is one flower in the
    # image.
    return 1 - discrete_classes


def count_with_patches(
    model: tf.keras.Model,
    images: tf.Tensor,
    *,
    patch_scale: float,
    patch_stride: Optional[float] = None,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Uses overlapping patches to estimate the total count in an image.

    Args:
        model: The model to use for inference.
        images: The images to process, as a 4D tensor.
        patch_scale: The patch scaling factor to use for patch extraction.
        patch_stride: The stride to use for patch extraction. If not
            specified, it defaults to 1/2 the patch scale.
        batch_size: The batch size to use for inference.

    Returns:
        An approximate density maps for the images, as a 4D array.

    """
    if patch_stride is None:
        # Use the default stride.
        patch_stride = patch_scale / 2.0

    logger.info("Extracting patches and running inference.")
    patches = extract_standard_patches(
        images, patch_scale=patch_scale, patch_stride=patch_stride
    )
    predictions = model.predict(patches, batch_size=batch_size)
    predicted_counts = _classes_to_counts(predictions)

    image_batch_shape = tf.shape(images).numpy()
    logger.debug("Input images have shape {}.", image_batch_shape)
    image_shape = image_batch_shape[1:3]
    num_images = image_batch_shape[0]
    num_patches = tf.shape(patches).numpy()[0]
    num_patches_per_image = num_patches // num_images
    logger.debug("Have {} patches per image.", num_patches_per_image)

    # These remain constant for a given scale and stride.
    patch_indices = _make_patch_indices(
        image_shape=image_shape,
        patch_scale=patch_scale,
        patch_stride=patch_stride,
    )
    patch_overlap_map = _make_patch_overlap_map(
        patch_indices=patch_indices, image_shape=image_shape
    )

    # Calculate density maps.
    density_maps = []
    for i in range(0, num_patches, num_patches_per_image):
        patch_predictions = predicted_counts[i : i + num_patches_per_image]
        density_maps.append(
            _make_density_map(
                patch_predictions=patch_predictions,
                patch_indices=patch_indices,
                patch_overlap_map=patch_overlap_map,
            )
        )

    return np.stack(density_maps, axis=0)
