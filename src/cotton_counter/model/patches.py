"""
Utilities for handling input patching.
"""


from multiprocessing import cpu_count
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

_NUM_THREADS = cpu_count()


def _crop_image_batch(
    images: tf.Tensor, *, corner_points: tf.Tensor, patch_scale: float
) -> tf.Tensor:
    """
    Crops a batch of images, with a different crop specified for each image
    in the batch.

    Args:
        images: The batch of images, of the shape (N, height, width, channels).
        corner_points: The top left corner points for each crop, of the shape
            (N, 2).
        patch_scale: The scaling factor to use for the extracted patches. (It
            will maintain the same aspect ratio.)

    Returns:
        The extracted crops.

    """
    image_shape = tf.shape(images)
    batch_shape = image_shape[0]
    single_image_shape = image_shape[1:3]

    # Find the patch size in pixels.
    patch_shape_pixels = tf.cast(single_image_shape, tf.float32) * tf.constant(
        patch_scale
    )
    patch_shape_pixels = tf.cast(patch_shape_pixels, tf.int32)

    # Acceptable pixel values for the corner points are chosen to make sure that
    # the final patches aren't out-of-bounds.
    max_corner_pixels = single_image_shape - patch_shape_pixels
    # Convert to pixels.
    corner_points_pixels = corner_points * tf.cast(
        max_corner_pixels, tf.float32
    )
    corner_points_pixels = tf.cast(corner_points_pixels, tf.int32)

    offset_height_px = corner_points_pixels[:, 0]
    offset_width_px = corner_points_pixels[:, 1]
    # Expand the patch shape so it is usable with map_fn.
    crop_height_px = tf.tile([patch_shape_pixels[0]], [batch_shape])
    crop_width_px = tf.tile([patch_shape_pixels[1]], [batch_shape])

    # Crop the images.
    return tf.map_fn(
        lambda args: tf.image.crop_to_bounding_box(*args),
        (
            images,
            offset_height_px,
            offset_width_px,
            crop_height_px,
            crop_width_px,
        ),
        dtype=images.dtype,
        parallel_iterations=_NUM_THREADS,
    )


def extract_random_patches(
    *, images: tf.Tensor, density_maps: tf.Tensor, patch_scale: float
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Extracts random patches from an image batch with the equivalent random
    patches the density maps.

    Args:
        images: The batch of images to extract the patch from.
        density_maps: The corresponding batch of density maps.
        patch_scale: The scaling factor to use for the extracted patches. (It
            will maintain the same aspect ratio.)

    Returns:
        The extracted patches from the image and density map.

    """
    image_shape = tf.shape(images)
    batch_size = image_shape[0]

    # Determine corner points for the patches in terms of frame fractions.
    corner_points = tf.random.uniform((batch_size, 2), name="random_patch")

    # Perform the crops.
    image_patches = _crop_image_batch(
        images, corner_points=corner_points, patch_scale=patch_scale
    )
    density_patches = _crop_image_batch(
        density_maps, corner_points=corner_points, patch_scale=patch_scale
    )
    return image_patches, density_patches


def extract_standard_patches(
    images: tf.Tensor, *, patch_scale: float, patch_stride: float,
) -> tf.Tensor:
    """
    Extracts a number of patches from each image in the input.

    Args:
        images: The images to extract patches from.
        patch_scale: The scale factor for each patch, in frame fractions.
        patch_stride: The stride for each patch, in frame fractions.

    Returns:
        A batch of all extracted patches.

    """
    # A static shape for the images must be available to use extract_patches.
    image_shape = images.shape[1:]
    if None in image_shape:
        # We need this shape to be known at compile time.
        raise ValueError(
            f"The shape of the images to extract patches from "
            f"must be known statically, but it is {image_shape}."
        )
    image_shape = np.array(image_shape)
    image_size = image_shape[0:2]
    image_channels = image_shape[2]

    # Convert to pixels.
    patch_scale_px = image_size * patch_scale
    patch_scale_px = patch_scale_px.astype(np.int32)
    patch_stride_px = image_size * patch_stride
    patch_stride_px = patch_stride_px.astype(np.int32)

    # Put into a form that we can use for patch extraction.
    kernel_size = [1] + patch_scale_px.tolist() + [1]
    kernel_strides = [1] + patch_stride_px.tolist() + [1]
    # Perform the extraction.
    flat_patches = tf.image.extract_patches(
        images=images,
        sizes=kernel_size,
        strides=kernel_strides,
        rates=[1, 1, 1, 1],
        padding="VALID",
    )

    # Collapse the first three dimensions into a single index.
    num_patch_pixels = tf.shape(flat_patches)[-1]
    flat_patches = tf.reshape(flat_patches, (-1, num_patch_pixels))
    # The patches are flattened in the last dimension, so as a last step we
    # expand them into their own batch.
    return tf.reshape(
        flat_patches,
        (-1, patch_scale_px[0], patch_scale_px[1], image_channels),
    )


def extract_standard_patches_from_dataset(
    *,
    images: tf.Tensor,
    density_maps: tf.Tensor,
    patch_scale: float,
    patch_stride: Optional[float] = None,
) -> tf.data.Dataset:
    """
    Extracts a standard set of patches from an image batch with the
    equivalent patches also extracted from the density maps. This is meant to
    be used for evaluation so we can make a direct comparison between
    different runs.

    This function is provided mostly as a convenience wrapper around
    `extract_standard_patches`, and is meant to be used as a target for
    `Dataset.flat_map()` or `Dataset.interleave()`.

    Args:
        images: The batch of images to extract patches from. Note that `images`
            must have a static shape defined for all dimensions except the
            batch.
        density_maps: The corresponding batch of density maps. Note that
            `density_maps` must have a static shape defined for all
            dimensions except the batch.
        patch_scale: The scaling factor to use for the extracted patches. (It
            will maintain the same aspect ratio.)
        patch_stride: The stride to use when sampling patches. By default,
            this is the same as the scale, but if you want overlapping
            or dilated patches, you can set it to a different number.

    Returns:
        The extracted patches from the image and density map, as a new Dataset.
        Each one is a 4D tensor where the batch dimension may be larger than
        that of the input.

    """
    if patch_stride is None:
        # Use non-overlapping patches by default.
        patch_stride = patch_scale

    image_patches = extract_standard_patches(
        images, patch_scale=patch_scale, patch_stride=patch_stride
    )
    density_patches = extract_standard_patches(
        density_maps, patch_scale=patch_scale, patch_stride=patch_stride
    )

    return tf.data.Dataset.from_tensor_slices((image_patches, density_patches))
