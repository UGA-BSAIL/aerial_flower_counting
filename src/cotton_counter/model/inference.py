"""
Helpers for performing inference.
"""

from typing import Dict, Iterable, Optional

import numpy as np
import tensorflow as tf
from loguru import logger

from src.cotton_counter.type_helpers import Vector2I

from . import patches


def _make_patch_indices_absolute(
    *, image_size: Vector2I, sizes: Iterable[int], strides: Iterable[int]
) -> np.ndarray:
    """
    Calculates the indices in the flattened input image that correspond to
    each extracted patch for a particular patch size and stride.

    Args:
        image_size: The size of the input image, in the form (height, width).
        sizes: The sizes argument that will be passed to
            `tf.image.extract_patches`.
        strides: The strides argument that will be passed to
            `tf.image.extract_patches`.

    Returns:
        An array, where the rows correspond to the indices in each patch.

    """
    # Create the 2D index array.
    image_size_flat = np.prod(image_size)
    index_array = np.arange(0, image_size_flat)
    index_array = index_array.reshape(image_size)

    # Extract the same patches from the index array.
    index_array_4d = np.expand_dims(index_array, axis=0)
    index_array_4d = np.expand_dims(index_array_4d, axis=3)
    index_array_4d = tf.constant(index_array_4d, dtype=tf.int32)
    index_patches = tf.image.extract_patches(
        images=index_array_4d,
        sizes=sizes,
        strides=strides,
        rates=[1, 1, 1, 1],
        padding="SAME",
    )

    # Flatten them into index arrays.
    index_patches = index_patches.numpy()
    # The total number of patches is specified by the second and third
    # dimensions. (This first batch dimension should always be one.)
    num_patches = np.prod(index_patches.shape[:3])
    return index_patches.reshape((num_patches, -1))


def _make_patch_indices_relative(
    *, image_size: Vector2I, patch_scale: float, patch_stride: float
) -> np.ndarray:
    """
    Calculates the indices in the flattened input image that correspond to
    each extracted patch for a particular patch scale and stride.

    This is different from `_make_patch_indices_absolute` in that it uses a
    relative patch stride and scale.

    Args:
        image_size: The size of the input image, in the form (height, width).
        patch_scale: The scale of the patches to extract, between 0.0 and 1.0.
        patch_stride: The stride of the patches to extract, between 0.0 and 1.0.

    Returns:
        An array, where the rows correspond to the indices in each patch.

    """
    kernel_size = patches.frame_fraction_to_size_or_stride(
        patch_scale, image_size=image_size
    )
    kernel_strides = patches.frame_fraction_to_size_or_stride(
        patch_stride, image_size=image_size
    )

    return _make_patch_indices_absolute(
        image_size=image_size, sizes=kernel_size, strides=kernel_strides
    )


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
        A 2D integer array of the same shape as `image_size`.

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
        patch_size = len(indices)
        # Distribute the density evenly over the entire patch.
        counts_per_pixel[indices] += prediction / patch_size

    # Compute the density map.
    counts_per_pixel = counts_per_pixel.reshape(patch_overlap_map.shape)
    density_map = counts_per_pixel / patch_overlap_map

    return np.expand_dims(density_map, axis=2)


def _classes_to_counts(y_hat: np.ndarray) -> np.ndarray:
    """
    Converts raw predictions from the model to count estimates.

    Args:
        y_hat: The raw predictions from the model.

    Returns:
        The estimated counts.

    """
    y_hat = y_hat.squeeze()
    # Convert from the sigmoid distribution to actual classes.
    discrete_classes = y_hat > 0.5
    discrete_classes = discrete_classes.astype(np.int32)

    # In this case, we assume that class zero means there is one flower in the
    # image.
    return 1 - discrete_classes


def _predict_with_activation_maps(
    model: tf.keras.Model, images: tf.Tensor, *, batch_size: int
) -> tf.Tensor:
    """
    Predicts with the model on a set of input images and returns the
    corresponding activation maps.

    Args:
        model:
        images:
        batch_size:

    Returns:

    """


def _extract_complete_activation_maps(
    model: tf.keras.Model,
    images: tf.Tensor,
    *,
    image_size: Vector2I,
    batch_size: int,
) -> tf.Tensor:
    """
    Extracts the complete activation map for a large input image. It will
    actually perform the inference in patches, so as not to use too much memory.

    Notes:
        The activation map will be resized to be the same size as the input,
        meaning that there will always be a pixel-to-pixel correspondence
        between the input and activation map regardless of how much the model
        down-samples.

    Args:
        model: The model to use for generating the activation map. Note that
            it is imperative that model's input size evenly divides the image
            size.
        images: The images to generate activation maps for.
        image_size: The size of the input images, in the form (height, width).
        batch_size: Batch size to use when extracting activation maps.

    Returns:
        The extracted activation maps, in a single Tensor.

    """
    model_input_size = model.input_shape[1:3]
    model_input_channels = model.input_shape[3]
    logger.debug(
        "Model has input size {}.", model_input_size,
    )

    # Extract patches that are the same shape as we specified for the model
    # input.
    kernel_size = [1] + list(model_input_size) + [1]
    flat_patches = tf.image.extract_patches(
        images=images,
        sizes=kernel_size,
        strides=kernel_size,
        rates=[1, 1, 1, 1],
        padding="SAME",
    )
    # The patches are flattened in the last dimension. Expand them into their
    # own batch.
    got_patches = tf.reshape(
        flat_patches,
        (-1, model_input_size[0], model_input_size[1], model_input_channels),
    )

    # Modify the model to produce activation maps.
    activation_layer = model.get_layer("activation_maps")
    activation_output = activation_layer.get_output_at(0)
    activation_model = tf.keras.Model(
        inputs=model.inputs, outputs=[activation_output]
    )

    # Apply the model to produce activation maps for the patches.
    logger.info("Producing activation maps...")
    activation_patches = activation_model.predict(
        got_patches, batch_size=batch_size
    )
    # Resize patches to be the same as the input.
    activation_patches = tf.image.resize(activation_patches, model_input_size)

    patch_indices = _make_patch_indices_absolute(
        image_size=image_size, sizes=kernel_size, strides=kernel_size
    )
    # _make_patch_indices_absolute gives us the indices in the original image
    # that correspond to the indices in the patches. However, we need the
    # other way around, so we get this through a scatter operation.
    counter = tf.range(patch_indices.size)
    gather_indices = tf.scatter_nd(
        tf.reshape(patch_indices, (-1, 1)),
        counter,
        shape=(patch_indices.size,),
    )
    # Re-combine patches into a single image.
    num_images = tf.shape(images)[0]
    activation_maps_flat = tf.gather(
        tf.reshape(activation_patches, (num_images, -1)),
        gather_indices,
        axis=1,
    )

    # The activation maps have only one channel.
    return tf.expand_dims(
        tf.reshape(activation_maps_flat, tf.shape(images)[:3]), axis=-1
    )


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
        Approximate density maps for the images, as a 4D array.

    """
    if patch_stride is None:
        # Use the default stride.
        patch_stride = patch_scale / 2.0

    image_batch_shape = tf.shape(images).numpy()
    logger.debug("Input images have shape {}.", image_batch_shape)
    image_size = image_batch_shape[1:3]
    num_images = image_batch_shape[0]

    # Get the activation maps for these images.
    activation_maps = _extract_complete_activation_maps(
        model, images, image_size=image_size, batch_size=batch_size
    )

    # Extract patches from the activation maps and produce actual counts.
    got_patches = patches.extract_standard_patches(
        activation_maps, patch_scale=patch_scale, patch_stride=patch_stride
    )
    predictions = tf.nn.sigmoid(tf.reduce_mean(got_patches, axis=(1, 2, 3)))
    predicted_counts = _classes_to_counts(predictions.numpy())

    num_patches = tf.shape(got_patches).numpy()[0]
    num_patches_per_image = num_patches // num_images
    logger.debug("Have {} patches per image.", num_patches_per_image)

    # These remain constant for a given scale and stride.
    patch_indices = _make_patch_indices_relative(
        image_size=image_size,
        patch_scale=patch_scale,
        patch_stride=patch_stride,
    )
    patch_overlap_map = _make_patch_overlap_map(
        patch_indices=patch_indices, image_shape=image_size
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


def calculate_max_density(images: tf.Tensor, *, patch_scale: float) -> float:
    """
    Calculates the maximum value that can appear in a density map for a certain
    image and patch scale.

    Args:
        images: The images to calculate this for.
        patch_scale: The scale of the patches to extract.

    Returns:
        The maximum value that can appear in a density map.

    """
    # Calculate the patch shape.
    image_batch_shape = tf.shape(images).numpy()
    single_image_shape = image_batch_shape[1:3]
    patch_shape = single_image_shape * patch_scale
    logger.debug("Calculated patch shape of {}.", patch_shape)

    # The maximum density is just the reciprocal, because at most we can have
    # one flower per patch.
    return 1.0 / np.prod(patch_shape)
