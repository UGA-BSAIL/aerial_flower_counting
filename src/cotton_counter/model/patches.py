"""
Utilities for handling input patching.
"""


from multiprocessing import cpu_count
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf

from ..type_helpers import Vector2I
from .records import Annotations

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
            (N, 2). The first column is y, and the second is x.
        patch_scale: The scaling factor to use for the extracted patches. (It
            will maintain the same aspect ratio.)

    Returns:
        The extracted crops.

    """
    tf.debugging.assert_less_equal(
        corner_points + patch_scale,
        1.0,
        message="Selected crop is out-of-bounds",
    )

    image_shape = tf.shape(images)
    batch_shape = image_shape[0]
    single_image_shape = image_shape[1:3]

    # Find the patch size in pixels.
    patch_shape_pixels = tf.cast(single_image_shape, tf.float32) * tf.constant(
        patch_scale
    )
    patch_shape_pixels = tf.cast(patch_shape_pixels, tf.int32)

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
        parallel_iterations=_NUM_THREADS,
    )


def _get_patch_coordinates(
    *, image_size: Vector2I, patch_scale: float, patch_stride: float
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Calculates the vertical and horizontal pixel coordinates in an image that
    correspond to each patch we want to extract.

    Args:
        image_size: The size of the input image, in the form (height, width).
        patch_scale: The patch scale to use.
        patch_stride: The patch stride to use.

    Returns:
        Two 2D tensors, for the vertical and horizontal coordinates,
        respectively. Each row in each tensor contains the coordinates for a
        single patch.

    """
    # Create the 2D index arrays for the vertical and horizontal dimensions.
    image_height, image_width = image_size

    vertical_indices = np.arange(0, image_height)
    vertical_index_array = np.repeat(
        np.expand_dims(vertical_indices, axis=1), image_width, axis=1
    )

    horizontal_indices = np.arange(0, image_width)
    horizontal_index_array = np.repeat(
        np.expand_dims(horizontal_indices, axis=0), image_height, axis=0
    )

    def extract_patch_indices(index_array: np.ndarray) -> tf.Tensor:
        """
        Extracts and flattens patches from an index array to compute the indices
        that make up each patch.

        Args:
            index_array: The index array to extract from.

        Returns:
            2D tensor where the rows correspond to the indices in each patch.

        """
        index_array_4d = np.expand_dims(index_array, axis=0)
        index_array_4d = np.expand_dims(index_array_4d, axis=3)
        index_array_4d = tf.constant(index_array_4d, dtype=tf.int32)
        index_patches = extract_standard_patches(
            index_array_4d, patch_scale=patch_scale, patch_stride=patch_stride
        )

        # Flatten them into index arrays.
        num_patches = tf.shape(index_patches)[0]
        return tf.reshape(index_patches, (num_patches, -1))

    vertical_patch_indices = extract_patch_indices(vertical_index_array)
    horizontal_patch_indices = extract_patch_indices(horizontal_index_array)
    return vertical_patch_indices, horizontal_patch_indices


def _make_patch_bboxes(
    *,
    image_shape: tf.Tensor,
    vertical_patch_coords: tf.Tensor,
    horizontal_patch_coords: tf.Tensor,
) -> tf.Tensor:
    """
    Creates bounding boxes for each patch.

    Args:
        image_shape: The 3D shape of the raw input images.
            (height, width, channels).
        vertical_patch_coords: The vertical coordinates in the input image
            for each patch.
        horizontal_patch_coords: The horizontal coordinates in the input
            image for each patch.

    Returns:
        An 2D tensor of bounding boxes where each row is of the form
        (min_y, min_x, max_y, max_x). All coordinates will be in frame
        fractions.

    """
    # Compute the bounding values for each patch.
    min_y_px = vertical_patch_coords[:, 0]
    max_y_px = vertical_patch_coords[:, -1]
    min_x_px = horizontal_patch_coords[:, 0]
    max_x_px = horizontal_patch_coords[:, -1]

    bounding_box_px = tf.stack(
        (min_y_px, min_x_px, max_y_px, max_x_px), axis=1
    )

    # Resulting vector should be (height, width, height, width).
    image_shape_expanded = tf.tile(image_shape[0:2], (2,))
    # Convert to frame fractions, like the annotations.
    return tf.cast(bounding_box_px, tf.float32) / tf.cast(
        image_shape_expanded, tf.float32
    )


def _get_points_within_boxes(
    *, points: tf.Tensor, bounding_boxes: tf.Tensor,
) -> tf.RaggedTensor:
    """
    Extracts only the 2D points from an array that are within a specific
    bounding box.

    Args:
        points: The points to filter. Should be a 2D array where each row is
            of the form (y, x).
        bounding_boxes: The bounding boxes to filter to. Should be a 2D array,
            where each row is of the form (min_y, min_x, max_y, max_x).

    Returns:
        A version of `points` where rows that fall outside the bounding box
        have been removed. It will also have an extra leading dimension for
        each individual bounding box that was specified.

    """
    tf.debugging.assert_shapes(
        [(points, (None, 2)), (bounding_boxes, (None, 4))],
        message="points or bounding_boxes have invalid shape",
    )

    # Add an extra leading dimension to points to match the bounding boxes.
    points = tf.expand_dims(points, axis=0)
    points = tf.repeat(points, tf.shape(bounding_boxes)[0], axis=0)

    # Make masks for all conditions.
    bounding_boxes = tf.expand_dims(bounding_boxes, axis=1)
    x_y_not_too_low = points >= bounding_boxes[:, :, 0:2]
    x_y_not_too_high = points <= bounding_boxes[:, :, 2:4]

    # Convert to vectors, where all coordinates have to be valid for the
    # whole point to be valid.
    x_y_not_too_low = tf.reduce_all(x_y_not_too_low, axis=-1)
    x_y_not_too_high = tf.reduce_all(x_y_not_too_high, axis=-1)
    # Finally, the points are only valid if all conditions are true.
    points_valid = tf.math.logical_and(x_y_not_too_low, x_y_not_too_high)

    # Mask out the invalid points.
    return tf.ragged.boolean_mask(points, points_valid)


def _get_annotations_in_boxes(
    *,
    annotations: Annotations,
    bounding_boxes: tf.Tensor,
    single_box_per_image: bool = False,
) -> Annotations:
    """
    For each specified bounding box, extracts all the annotations that are
    within that box.

    Args:
        annotations: The annotations to filter.
        bounding_boxes: The bounding boxes to filter to. Should be a 2D array,
            where each row is of the form (min_y, min_x, max_y, max_x).
        single_box_per_image: This controls how it handles bounding boxes. If
            this is true, then the outer dimension of `bounding_boxes` should
            be the same size as the batch dimension of the annotations, and it
            will interpret each bounding box as corresponding to a single item
            in the batch. Otherwise, it will extract all bounding boxes for
            every item in the batch.

    Returns:
        Similar annotations, but filtered to only contain the annotations that
        lie within each bounding box. The dimensions of each element in the
        `Annotations` structure will be (batch, point), where the last
        dimension is ragged. The ordering of the batch dimension will be
        the same as the ordering of the input annotations, with all data
        from each bounding box grouped contiguously. Note
        also that frame numbers will be ignored, and not included in the output.

    """

    def _filter_one_frame(
        *, y_values: tf.Tensor, x_values: tf.Tensor, boxes: tf.Tensor,
    ) -> Tuple[tf.RaggedTensor, tf.RaggedTensor]:
        """
        Filters annotations for a single frame.

        Args:
            y_values: The annotation Y values to filter.
            x_values: The annotation X values to filter.
            boxes: The corresponding bounding boxes to use for filtering.

        Returns:
            The filtered x and y coordinates of the annotations for the frame.

        """
        # Combine to a single array.
        points = tf.stack((y_values, x_values), axis=1)
        filtered = _get_points_within_boxes(
            points=points, bounding_boxes=boxes
        )

        # Split the output again.
        return filtered[:, :, 1], filtered[:, :, 0]

    def _filter_one_frame_single_box(
        x_y_and_box: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
    ) -> Tuple[tf.RaggedTensor, tf.RaggedTensor]:
        """
        Filters annotations for a single frame using one bounding box per frame.

        Args:
            x_y_and_box: Tuple containing annotation x-coordinates,
                y-coordinates, and the 1D bounding box to use.

        Returns:
            The filtered x and y coordinates of the annotations for the frame.

        """
        x_values, y_values, bounding_box = x_y_and_box

        tf.debugging.assert_shapes(
            [(bounding_box, (4,))], message="bounding_box"
        )
        # Add an extra dimension to the bounding box so we can use it for
        # extraction.
        bounding_box = tf.expand_dims(bounding_box, axis=0)

        return _filter_one_frame(
            x_values=x_values, y_values=y_values, boxes=bounding_box
        )

    def _filter_one_frame_multi_box(
        x_and_y: Tuple[tf.Tensor, tf.Tensor]
    ) -> Tuple[tf.RaggedTensor, tf.RaggedTensor]:
        """
        Filters annotations for a single frame using all bounding boxes per
        frame.

        Args:
            x_and_y: Tuple containing annotation x-coordinates and
                y-coordinates.

        Returns:
            The filtered x and y coordinates of the frame.

        """
        x_values, y_values = x_and_y
        return _filter_one_frame(
            x_values=x_values, y_values=y_values, boxes=bounding_boxes
        )

    map_func = _filter_one_frame_multi_box
    map_args = (annotations.x_values, annotations.y_values)
    if single_box_per_image:
        map_func = _filter_one_frame_single_box
        # Pass a single bounding box for each image.
        map_args += (bounding_boxes,)

    filtered_x, filtered_y = tf.map_fn(
        map_func,
        map_args,
        # The output from this function will have dimensions
        # (box, point)
        fn_output_signature=(tf.RaggedTensorSpec((None, None)),) * 2,
        parallel_iterations=_NUM_THREADS,
    )

    # Overall output has dimensions (frame, box, point). However, in order to
    # mirror the images, we want to merge the first two dimensions.
    filtered_y = filtered_y.merge_dims(0, 1)
    filtered_x = filtered_x.merge_dims(0, 1)

    return Annotations(x_values=filtered_x, y_values=filtered_y)


def _static_image_shape(images: tf.Tensor) -> np.ndarray:
    """
    Obtains the static image shape, verifying that it exists.

    Args:
        images: The images to extract the shape from.

    Returns:
        The 3D shape of a single image in the batch.

    """
    image_shape = images.shape[1:]
    if None in image_shape:
        # We need this shape to be known at compile time.
        raise ValueError(
            f"The shape of the images to extract patches from "
            f"must be known statically, but it is {image_shape}."
        )
    return np.array(image_shape)


def _make_dataset(
    images: tf.Tensor, annotations: Annotations
) -> tf.data.Dataset:
    """
    Basic helper function that creates a `Dataset` from a set of images and
    corresponding annotations.

    Args:
        images: The images to include in the dataset.
        annotations: The corresponding annotations.

    Returns:
        The `Dataset` that it created.

    """
    return tf.data.Dataset.from_tensor_slices(
        dict(
            image=images,
            annotation_x=annotations.x_values,
            annotation_y=annotations.y_values,
        )
    )


def frame_fraction_to_size_or_stride(
    frame_fraction: float, *, image_size: Tuple[int, int]
) -> List[int]:
    """
    Utility function that converts a patch scale or patch stride, expressed in
    frame fractions, to a kernel size/stride that that is compatible with
    `tf.image.extract_patches`.

    Args:
        frame_fraction: The patch scale or stride to convert, as a fraction
            of the input size.
        image_size: The size of the input image, in the form (height, width).

    Returns:
        The scale or stride expressed in a form that is compatible with
        `extract_patches`.

    """
    size_px = np.array(image_size) * frame_fraction
    size_px = size_px.astype(np.int32)
    return [1] + size_px.tolist() + [1]


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
        A batch of all extracted patches. They will be in the same order as the
        input, with the patches each image in the batch grouped together.

    """
    image_shape = _static_image_shape(images)
    image_size = image_shape[0:2]
    image_channels = image_shape[2]

    # Convert to pixels.
    kernel_size = frame_fraction_to_size_or_stride(
        patch_scale, image_size=image_size
    )
    kernel_strides = frame_fraction_to_size_or_stride(
        patch_stride, image_size=image_size
    )

    # Perform the extraction.
    flat_patches = tf.image.extract_patches(
        images=images,
        sizes=kernel_size,
        strides=kernel_strides,
        rates=[1, 1, 1, 1],
        padding="SAME",
    )

    # Collapse the first three dimensions into a single index.
    num_patch_pixels = tf.shape(flat_patches)[-1]
    flat_patches = tf.reshape(flat_patches, (-1, num_patch_pixels))
    # The patches are flattened in the last dimension, so as a last step we
    # expand them into their own batch.
    return tf.reshape(
        flat_patches, (-1, kernel_size[1], kernel_size[2], image_channels),
    )


def extract_random_annotated_patches(
    *, images: tf.Tensor, annotations: Annotations, patch_scale: float
) -> tf.data.Dataset:
    """
    Extracts random patches from an image batch with corresponding annotations.

    Args:
        images: The batch of images to extract the patch from.
        annotations: The corresponding annotations.
        patch_scale: The scaling factor to use for the extracted patches. (It
            will maintain the same aspect ratio.)

    Returns:
        The extracted patches from the image and corresponding annotations, as
        a `Dataset`.

    """
    image_shape = tf.shape(images)
    batch_size = image_shape[0]

    # Determine corner points for the patches in terms of frame fractions.
    corner_points = tf.random.uniform((batch_size, 2), name="random_patch")
    # Make sure the selected corner points remain in-bounds.
    corner_points *= 1.0 - patch_scale

    # Perform the crops.
    image_patches = _crop_image_batch(
        images, corner_points=corner_points, patch_scale=patch_scale
    )

    # Create a proper bounding box and crop the annotations.
    bounding_boxes = tf.concat(
        (corner_points, corner_points + patch_scale), axis=1
    )
    annotations_within = _get_annotations_in_boxes(
        annotations=annotations,
        bounding_boxes=bounding_boxes,
        single_box_per_image=True,
    )

    return _make_dataset(image_patches, annotations_within)


def extract_standard_annotated_patches(
    *,
    images: tf.Tensor,
    annotations: Annotations,
    patch_scale: float,
    patch_stride: Optional[float] = None,
) -> tf.data.Dataset:
    """
    Extracts a standard set of patches from an image batch with the
    corresponding annotations. This is meant to be used for evaluation so we
    can make a direct comparison between different runs.

    Args:
        images: The batch of images to extract patches from. Note that `images`
            must have a static shape defined for all dimensions except the
            batch.
        annotations: The corresponding annotations for the image. The
            annotations that fall within this patch will be extracted.
        patch_scale: The scaling factor to use for the extracted patches. (It
            will maintain the same aspect ratio.)
        patch_stride: The stride to use when sampling patches. By default,
            this is the same as the scale, but if you want overlapping
            or dilated patches, you can set it to a different number.

    Returns:
        The extracted patches from the image and corresponding annotations, as
        a `Dataset`.

    """
    if patch_stride is None:
        # Use non-overlapping patches by default.
        patch_stride = patch_scale

    image_patches = extract_standard_patches(
        images, patch_scale=patch_scale, patch_stride=patch_stride
    )

    # Extract the annotations.
    image_shape = _static_image_shape(images)
    vertical_coords, horizontal_coords = _get_patch_coordinates(
        image_size=tuple(image_shape.tolist())[0:2],
        patch_scale=patch_scale,
        patch_stride=patch_stride,
    )
    patch_bboxes = _make_patch_bboxes(
        image_shape=tf.constant(image_shape),
        vertical_patch_coords=vertical_coords,
        horizontal_patch_coords=horizontal_coords,
    )
    annotations_within = _get_annotations_in_boxes(
        annotations=annotations, bounding_boxes=patch_bboxes
    )

    return _make_dataset(image_patches, annotations_within)
