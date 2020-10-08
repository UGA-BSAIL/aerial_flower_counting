"""
Layer that creates density maps from a set of annotations.
"""


from functools import singledispatch
from typing import Any, Iterable, NoReturn, Tuple, Union

import tensorflow as tf
from loguru import logger

from ..type_helpers import Vector2I
from .filters import gaussian_blur
from .records import Annotations


def _normalize_frame_numbers(
    frame_numbers: tf.Tensor, skip_indices: tf.Tensor = tf.constant([])
) -> tf.Tensor:
    """
    The frame numbers provided with the annotations are essentially arbitrary
    IDs. However, when creating density maps, we want to be able to use them
    as an index into a batch. Therefore, this function takes the raw frame
    numbers as input and converts them to sequential integers ranging from 0
    to the total number of frames - 1.

    Examples:
        ```
        _normalize_frame_numbers([2, 2, 6, 6, 6, 10, 8, 8])
        # Returns [0, 0, 1, 1, 1, 2, 3, 3]
        ```

    Args:
        frame_numbers: The raw frame numbers to normalize.
        skip_indices: An optional 1D tensor specifying a list of normalized
            values that should be skipped in our output. This is primarily
            useful for handling frames that have no annotations.

    Returns:
        The normalized frame numbers.

    """
    frame_numbers = tf.cast(frame_numbers, tf.int64)
    skip_indices = tf.cast(skip_indices, tf.int64)

    def map_to_normal(
        frame_index: tf.Tensor,
        normed_value: tf.Tensor,
        last_frame_num: tf.Tensor,
        normed_so_far: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Computes the next partial value for the normalized frame numbers list.

        Args:
            frame_index: The index in the raw frame numbers list we are
                currently reading.
            normed_value: The current normalized frame number we are using.
            last_frame_num: The raw frame number we saw on the previous
                iteration.
            normed_so_far: The partial list of normalized values.

        Returns:
            The arguments to call this function with on the next iteration.

        """
        next_frame_num = frame_numbers[frame_index]
        next_normed = tf.cond(
            next_frame_num == last_frame_num,
            true_fn=lambda: normed_value,
            false_fn=lambda: normed_value + tf.constant(1, dtype=tf.int64),
        )

        should_skip = tf.reduce_any(tf.equal(skip_indices, next_normed))
        return tf.cond(
            should_skip,
            # If this index should be skipped, we run another iteration of
            # the loop without modifying anything but the normed value.
            true_fn=lambda: (
                frame_index,
                next_normed,
                last_frame_num,
                normed_so_far,
            ),
            # Otherwise, we update the state variables as normal and move on
            # to the next frame number.
            false_fn=lambda: (
                frame_index + tf.constant(1),
                next_normed,
                next_frame_num,
                tf.concat([normed_so_far, [next_normed]], 0),
            ),
        )

    # Set initial values for the loop variables.
    frame_index_0 = tf.constant(0)
    normed_value_0 = tf.constant(-1, dtype=tf.int64)
    last_frame_num_0 = tf.constant(-1, dtype=tf.int64)
    normed_so_far_0 = tf.constant([], dtype=tf.int64)
    # Shape invariants are all standard, except normed_so_far, which is
    # allowed to grow.
    shape_invariants = (
        frame_index_0.shape,
        normed_value_0.shape,
        last_frame_num_0.shape,
        tf.TensorShape([None]),
    )

    num_annotations = tf.size(frame_numbers)

    _, _, _, normed_frame_nums = tf.while_loop(
        lambda i, _1, _2, _3: i < num_annotations,
        map_to_normal,
        (frame_index_0, normed_value_0, last_frame_num_0, normed_so_far_0),
        shape_invariants=shape_invariants,
    )
    return normed_frame_nums


def _flatten_annotations(
    annotation_vectors: Iterable[tf.RaggedTensor],
) -> Iterable[tf.Tensor]:
    """
    Annotations come in as a `RaggedTensor` with one dimension being the batch,
    and the second being the actual annotations. However, for our processing,
    it's easier to just flatten the whole thing into one vector.

    Args:
        annotation_vectors: The separate annotation vectors to flatten.

    Returns:
        Equivalent annotations, but with no batch dimensions.

    """
    # Deliberately not a generator so that Tensorflow AutoGraph can handle it.
    flat_annotations = []
    for annotations in annotation_vectors:
        flat_annotations.append(annotations.flat_values)

    return flat_annotations


@singledispatch
def _make_point_annotation_maps(
    x_values: Any,
    y_values: Any,
    frame_numbers: Any,
    *,
    map_shape: Vector2I,
    batch_size: tf.Tensor,
) -> NoReturn:
    """
    Converts a set of sparse point annotations into images where each
    annotation is notated by a pixel with the value 1. All other pixels will
    be zeroed.

    It works with both normal and ragged tensors as input. If the former,
    it expects the input to be 1D. If the latter, it expects it to be 2D,
    with the first dimension being the batch.

    Args:
        x_values: The x-coordinates of the annotations.
        y_values: The y-coordinates of the annotations.
        frame_numbers: The corresponding frame numbers of the annotations.
        map_shape: The shape of the output density maps to generate, in the
            form (height, width). The samples dimension will be inferred.
        batch_size: The size of the batch of images that we are creating
            annotations for.

    Returns:
        A corresponding tensor of dense annotation maps, of shape
        (samples, height, width, 1).

    """
    raise NotImplementedError(
        f"_make_point_annotation_maps() is not implemented for input of type "
        f"{type(x_values)}."
    )


@_make_point_annotation_maps.register
def _(
    x_values: tf.Tensor,
    y_values: tf.Tensor,
    frame_numbers: tf.Tensor,
    *,
    map_shape: Vector2I,
    batch_size: tf.Tensor,
) -> tf.Tensor:
    batch_size = tf.ensure_shape(batch_size, ())
    batch_size = tf.cast(batch_size, tf.int64)

    # Quantize the annotations to convert from frame fractions to actual
    # pixel values.
    map_height, map_width = map_shape
    pixel_annotations_x = x_values * tf.constant(
        map_width - 1, dtype=tf.dtypes.float32
    )
    pixel_annotations_y = y_values * tf.constant(
        map_height - 1, dtype=tf.dtypes.float32
    )
    pixel_annotations_x = tf.cast(pixel_annotations_x, tf.dtypes.int64)
    pixel_annotations_y = tf.cast(pixel_annotations_y, tf.dtypes.int64)
    # Combine into one matrix.
    pixel_annotations = tf.stack(
        [frame_numbers, pixel_annotations_y, pixel_annotations_x], axis=1
    )

    # Compute the total number of samples (frames).
    map_shape_with_samples = tf.stack(
        [batch_size, map_height, map_width], axis=0,
    )

    # Generate the output maps.
    num_non_zero_values = tf.shape(pixel_annotations)[0]
    non_zeros = tf.ones((num_non_zero_values,), dtype=tf.float32)
    sparse_maps = tf.SparseTensor(
        indices=pixel_annotations,
        values=non_zeros,
        dense_shape=map_shape_with_samples,
    )
    # Reorder indices to conform with sparse tensor conventions.
    sparse_maps = tf.sparse.reorder(sparse_maps)

    dense = tf.sparse.to_dense(
        # There might be duplicate indices, which we want to ignore.
        sparse_maps,
        default_value=0.0,
        validate_indices=False,
    )
    # Add a dummy channel dimension.
    return tf.expand_dims(dense, 3)


@_make_point_annotation_maps.register
def _(
    x_values: tf.RaggedTensor,
    y_values: tf.RaggedTensor,
    frame_numbers: tf.RaggedTensor,
    *,
    map_shape: Vector2I,
    batch_size: tf.Tensor,
) -> tf.Tensor:
    # Flatten the annotations.
    x_values, y_values, frame_numbers = _flatten_annotations(
        (x_values, y_values, frame_numbers)
    )

    # The frame numbers must be normalized going into this function. In order
    # for that to work, we need to specify which frames should be blank, which
    # should correspond to any row with no annotations.
    row_lengths = frame_numbers.row_lengths()
    empty_rows_mask = tf.equal(row_lengths, 0)
    empty_row_indices = tf.range(tf.size(empty_rows_mask))[empty_rows_mask]
    normed_frame_nums = _normalize_frame_numbers(
        frame_numbers, skip_indices=empty_row_indices
    )

    return _make_point_annotation_maps(
        x_values,
        y_values,
        normed_frame_nums,
        map_shape=map_shape,
        batch_size=batch_size,
    )


def make_density_map(
    annotations_x: tf.Tensor,
    annotations_y: tf.Tensor,
    *,
    map_shape: Vector2I,
    sigma: float,
) -> tf.Tensor:
    """
    Creates a set of density maps from a set of annotations for a single frame.

    Args:
        annotations_x: A 1D tensor of the x-values of the annotation points.
        annotations_y: A 1D tensor of the y-values of the annotation points.
        map_shape: The shape of the output density maps to generate, in the
            form (height, width). The samples dimension will be inferred.
        sigma:
            The standard deviation in pixels to use for the applied gaussian
            filter.

    Returns:
        A tensor containing density maps, of the shape
        (height, width, 1)

    """
    # Our frame numbers are all going to be the same, since we only have one
    # image.
    frame_numbers = tf.zeros_like(annotations_x, dtype=tf.int64)

    annotations = Annotations(
        x_values=annotations_x,
        y_values=annotations_y,
        frame_numbers=frame_numbers,
    )
    return make_density_maps_batch(
        annotations, map_shape=map_shape, sigma=sigma, batch_size=1
    )[0]


def make_density_maps_batch(
    annotations: Annotations,
    *,
    map_shape: Vector2I,
    batch_size: Union[tf.Tensor, int],
    sigma: float,
) -> tf.Tensor:
    """
    Creates a set of density maps from a set of annotations for a batch of
    images.

    Args:
        annotations: The sparse annotations.
        map_shape: The shape of the output density maps to generate, in the
            form (height, width). The samples dimension will be inferred.
        batch_size: The size of the batch that we are creating annotations for.
        sigma:
            The standard deviation in pixels to use for the applied gaussian
            filter.

    Returns:
        A tensor containing density maps, of the shape
        (samples, height, width, 1)

    """
    batch_size = tf.convert_to_tensor(batch_size)

    # Compute our filter size so that it's odd and has sigma pixels on either
    # side.
    kernel_size = int(1 + 6 * sigma)
    logger.debug("Using {}-pixel kernel for gaussian blur.", kernel_size)

    # Obtain initial point annotations.
    dense_annotations = _make_point_annotation_maps(
        annotations.x_values,
        annotations.y_values,
        annotations.frame_numbers,
        map_shape=map_shape,
        batch_size=batch_size,
    )

    return gaussian_blur(
        dense_annotations, kernel_size=kernel_size, sigma=sigma,
    )
