"""
Utilities for handling input patching.
"""


import tensorflow as tf
from loguru import logger

from .records import Annotations


def _crop_image_batch(
    images: tf.Tensor, *, corner_points: tf.Tensor, patch_scale: float
) -> tf.Tensor:
    """
    Crops a batch of images, with a different crop specified for each image
    in the batch.

    Args:
        images: The batch of images, of the shape (N, height, width, channels).
        corner_points: The top left corner points for each crop, of the shape
            (N, 2). The column order should be x, y.
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

    # Corner points are in the form (w, h) so we reverse them to get (h, w).
    corner_points = corner_points[:, ::-1]
    # Convert to pixels.
    corner_points_pixels = corner_points * tf.cast(
        single_image_shape, tf.float32
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


def _get_annotations_in_box(
    annotations: Annotations, *, top_left: tf.Tensor, bottom_right: tf.Tensor,
) -> Annotations:
    """
    Extracts only the 2D points from an array that are within a specific
    bounding box.

    Args:
        annotations: The annotations that we are filtering.
        top_left: The top left point of the bounding box.
        bottom_right: The bottom right point of the bounding box.

    Returns:
        New annotations that are within the bounding box.

    """
    # Convert annotations to a matrix for easy comparison.
    points = tf.stack((annotations.x_values, annotations.y_values), axis=1)

    # Make masks for all conditions.
    x_y_not_too_low = points >= top_left
    x_y_not_too_high = points <= bottom_right

    # Convert to vectors, where all coordinates have to be valid for the
    # whole point to be valid.
    x_y_not_too_low = tf.reduce_all(x_y_not_too_low, axis=1)
    x_y_not_too_high = tf.reduce_all(x_y_not_too_high, axis=1)
    # Finally, the points are only valid if all conditions are true.
    points_valid = tf.math.logical_and(x_y_not_too_low, x_y_not_too_high)

    # Mask out the invalid points.
    new_x = tf.boolean_mask(annotations.x_values, points_valid)
    new_y = tf.boolean_mask(annotations.y_values, points_valid)
    new_frames = tf.boolean_mask(annotations.frame_numbers, points_valid)

    return Annotations(
        x_values=new_x, y_values=new_y, frame_numbers=new_frames
    )


def _extract_specific_patch(
    image: tf.Tensor,
    *,
    annotations: Annotations,
    top_left: tf.Tensor,
    patch_scale: float,
) -> tf.data.Dataset:
    """
    Extracts a specific patch from a single image.

    Args:
        image: The image to extract the patch from.
        annotations: The corresponding annotations for the image. The
            annotations that fall within this patch will be extracted.
        top_left: The top left point of the patch bounding box.
        patch_scale: The scaling factor to use for the extracted patches. (It
            will maintain the same aspect ratio.)

    Returns:
        The extracted patch from the image and the annotations that fall
        within it, as a new dataset. The new dataset will have an "image" key,
        containing the extracted patch, as well as "annotation_x",
        "annotation_y", and "frame_numbers" keys for the annotations.

    """
    # Extract only the annotations that are within the patches.
    annotations_within = _get_annotations_in_box(
        annotations,
        top_left=top_left[0],
        bottom_right=top_left[0] + patch_scale,
    )

    # Re-reference the annotations to the patch instead of the full image.
    annotations_within.x_values -= top_left[0][0]
    annotations_within.y_values -= top_left[0][1]
    annotations_within.x_values /= patch_scale
    annotations_within.y_values /= patch_scale

    # Perform the crop.
    image_expanded = tf.expand_dims(image, axis=0)
    image_patch = _crop_image_batch(
        image_expanded, corner_points=top_left, patch_scale=patch_scale
    )[0]
    return tf.data.Dataset.from_tensors(
        dict(
            image=image_patch,
            annotation_x=annotations_within.x_values,
            annotation_y=annotations_within.y_values,
            frame_numbers=annotations_within.frame_numbers,
        )
    )


def extract_random_patch(
    *, image: tf.Tensor, annotations: Annotations, patch_scale: float
) -> tf.data.Dataset:
    """
    Extracts a random patch from an image, as well as the corresponding
    annotations that fall within that patch.

    Args:
        image: The image to extract the patch from.
        annotations: The corresponding annotation points for the image.
        patch_scale: The scaling factor to use for the extracted patches. (It
            will maintain the same aspect ratio.)

    Returns:
        The extracted patch from the image and the annotations that fall
        within it, as a new dataset. The new dataset will have an "image" key,
        containing the extracted patch, as well as "annotation_x",
        "annotation_y", and "frame_numbers" keys for the annotations.

    """
    # Determine corner points for the patches in terms of frame fractions.
    top_left = tf.random.uniform((1, 2), name="random_patch")
    # Make sure we can't go off the edge of the frame.
    top_left *= 1.0 - patch_scale

    return _extract_specific_patch(
        image,
        annotations=annotations,
        top_left=top_left,
        patch_scale=patch_scale,
    )


def _cartesian_product(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """
    Computes the cartesian product of 2 1D tensors. Taken from this SO answer:
    https://stackoverflow.com/a/47133461

    Args:
        a: The first 1D tensor.
        b: The second 1D tensor.

    Returns:
        A 2D tensor containing the cartesian product.

    """
    tile_a = tf.tile(tf.expand_dims(a, 1), [1, tf.shape(b)[0]])
    tile_a = tf.expand_dims(tile_a, 2)
    tile_b = tf.tile(tf.expand_dims(b, 0), [tf.shape(a)[0], 1])
    tile_b = tf.expand_dims(tile_b, 2)

    product = tf.concat([tile_a, tile_b], axis=2)
    return tf.reshape(product, [-1, 2])


def extract_standard_patches(
    *, image: tf.Tensor, annotations: Annotations, patch_scale: float
) -> tf.data.Dataset:
    """
    Extracts a standard set of patches from an image batch with the
    equivalent patches also extracted from the density maps. This is meant to
    be used for evaluation so we can make a direct comparison between
    different runs.

    Args:
        image: The image to extract the patches from.
        annotations: The corresponding annotation points for the image.
        patch_scale: The scaling factor to use for the extracted patches. (It
            will maintain the same aspect ratio.)

    Returns:
        The extracted patches from the image and density map, as a new Dataset.
        Each one is a 4D tensor where the batch dimension may be larger than
        that of the input.

    """
    # Determine the patch spacing.
    num_patches_1_dim = int(1.0 / patch_scale)
    num_patches = num_patches_1_dim ** 2
    logger.debug("Using {} patches on each side.", num_patches_1_dim)

    spacing_1d = tf.linspace(
        tf.constant(0.0), tf.constant(1.0 - patch_scale), num_patches_1_dim
    )
    # Combine them to get patch coordinates in frame fractions.
    patch_coords = _cartesian_product(spacing_1d, spacing_1d)

    # Crop each patch from each image in the batch.
    patch_datasets = []
    for i in range(num_patches):
        this_patch_coords = tf.expand_dims(patch_coords[i], axis=0)

        patch_dataset = _extract_specific_patch(
            image,
            annotations=annotations,
            top_left=this_patch_coords,
            patch_scale=patch_scale,
        )
        patch_datasets.append(patch_dataset)

    # Combine the datasets for each patch into a single one.
    choices = tf.data.Dataset.range(len(patch_datasets))
    return tf.data.experimental.choose_from_datasets(patch_datasets, choices)
