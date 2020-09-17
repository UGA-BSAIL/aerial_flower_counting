"""
Utilities for handling input patching.
"""


from typing import Tuple

import tensorflow as tf


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
    )


def extract_random_patches(
    *, images: tf.Tensor, density_maps: tf.Tensor, patch_scale: float
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Extracts random patched from an image batch with the equivalent random
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
