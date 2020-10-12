"""
Extracts the information that we need to train the model from a raw
Tensorflow `Dataset`.
"""

import functools
from multiprocessing import cpu_count
from typing import Dict, List, Tuple

import tensorflow as tf

from ..type_helpers import Vector2I
from .density_maps import make_density_maps
from .patches import extract_random_patches, extract_standard_patches
from .records import Annotations

_FEATURE_DESCRIPTION = {
    "image": tf.io.FixedLenFeature([1], tf.dtypes.string),
    "frame_numbers": tf.io.VarLenFeature(tf.dtypes.int64),
    "annotation_x": tf.io.VarLenFeature(tf.dtypes.float32),
    "annotation_y": tf.io.VarLenFeature(tf.dtypes.float32),
}
"""
Descriptions of the features found in our dataset.
"""

_NUM_THREADS = cpu_count()


def _decode_jpeg(jpeg_batch: tf.Tensor) -> tf.Tensor:
    """
    Decodes JPEG images from a feature dictionary.

    Args:
        jpeg_batch: The batch of JPEG images.

    Returns:
        The 4D batch of decoded images.

    """
    # This is going to have a batch dimension, so we need to map it.
    return tf.map_fn(
        lambda j: tf.io.decode_jpeg(j[0]),
        jpeg_batch,
        dtype=tf.dtypes.uint8,
        parallel_iterations=_NUM_THREADS,
    )


def _parse_example(serialized: tf.Tensor) -> Dict[str, tf.Tensor]:
    """
    Deserializes a set of serialized examples.

    Args:
        serialized: The tensor containing a serialized example.


    Returns:
        The parsed feature dictionary.

    """
    return tf.io.parse_single_example(serialized, _FEATURE_DESCRIPTION)


def _load_from_feature_dict(
    feature_dict: Dict[str, tf.Tensor], *, map_shape: Vector2I, sigma: float
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Wrangles the data from a parsed feature dictionary into images and density
    maps.

    Args:
        feature_dict: The parsed feature dictionary.
        map_shape: The shape of the output density maps to generate, in the
            form (height, width). The samples dimension will be inferred.
        sigma:
            The standard deviation in pixels to use for the applied gaussian
            filter.

    Returns:
        The images from the batch as well as the generated density maps.

    """
    # Decode the JPEG images.
    images = _decode_jpeg(feature_dict["image"])

    # Convert from sparse to ragged tensors.
    frame_numbers = tf.RaggedTensor.from_sparse(feature_dict["frame_numbers"])
    x_values = tf.RaggedTensor.from_sparse(feature_dict["annotation_x"])
    y_values = tf.RaggedTensor.from_sparse(feature_dict["annotation_y"])

    # Create density maps.
    annotations = Annotations(
        frame_numbers=frame_numbers, x_values=x_values, y_values=y_values,
    )
    density_maps = make_density_maps(
        annotations, map_shape=map_shape, sigma=sigma
    )

    return images, density_maps


def _discretize_counts(
    counts: tf.Tensor, *, bucket_min_values: List[float]
) -> tf.Tensor:
    """
    Discretizes the count values into a fixed number of categories.

    Args:
        counts: The tensor containing raw count values.
        bucket_min_values: A list of the minimum values that will go in each
            bucket. Note that the highest bucket will contain anything that
            falls between the largest minimum value and infinity.

    Returns:
        An integer tensor containing the categorical counts.

    """
    # Make sure they're sorted from high to low.
    bucket_min_values.sort(reverse=True)

    # These masks will determine which counts fall in which buckets.
    bucket_masks = []
    for min_value in bucket_min_values:
        bucket_mask = counts >= tf.constant(min_value, dtype=tf.float32)
        bucket_masks.append(bucket_mask)

    # Generate the category vector.
    categorical_counts = tf.zeros_like(counts, dtype=tf.int32)
    already_filled = tf.zeros_like(categorical_counts, dtype=tf.bool)
    for i, bucket_mask in enumerate(bucket_masks):
        fill_mask = tf.logical_and(tf.logical_not(already_filled), bucket_mask)
        fill_mask_int = tf.cast(fill_mask, tf.int32)

        categorical_counts += fill_mask_int * tf.constant(i, dtype=tf.int32)

        already_filled = tf.logical_or(fill_mask, already_filled)

    return categorical_counts


def _add_counts(
    *,
    images: tf.Tensor,
    density_maps: tf.Tensor,
    bucket_min_values: List[float],
) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    """
    Adds summed count values to a `Dataset`.

    Args:
        images: The input image data.
        density_maps: The corresponding density maps.
        bucket_min_values: A list of the minimum count values that will go in
            each discrete count bucket. Note that the highest bucket will
            contain anything that falls between the largest minimum value and
            infinity.

    Returns:
        A dictionary mapping model inputs to tensors and a dictionary mapping
        model outputs to tensors.

    """
    counts = tf.reduce_sum(density_maps, axis=[1, 2, 3])

    # Discretize the counts.
    discrete_counts = _discretize_counts(
        counts, bucket_min_values=bucket_min_values
    )

    # Keras doesn't like 1D regression inputs, so add an extra dimension.
    counts = tf.expand_dims(counts, axis=1)

    return (
        dict(image=images),
        dict(
            density_map=density_maps,
            count=counts,
            discrete_count=discrete_counts,
        ),
    )


def extract_model_input(
    raw_dataset: tf.data.Dataset,
    *,
    map_shape: Vector2I,
    sigma: float,
    bucket_min_values: List[float],
    batch_size: int = 32,
    num_prefetch_batches: int = 5,
    patch_scale: float = 1.0,
    random_patches: bool = True,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """
    Deserializes raw data from a `Dataset`, and coerces it into the form
    used by the model.

    Args:
        raw_dataset: The raw dataset, containing serialized data.
        map_shape: The shape of the output density maps to generate, in the
            form (height, width). The samples dimension will be inferred.
        sigma:
            The standard deviation in pixels to use for the applied gaussian
            filter.
        bucket_min_values: A list of the minimum count values that will go in
            each discrete count bucket. Note that the highest bucket will
            contain anything that falls between the largest minimum value and
            infinity.
        batch_size: The size of the batches that we generate.
        num_prefetch_batches: The number of batches to prefetch into memory.
            Increasing this can increase performance at the expense of memory
            usage.
        patch_scale: Scale of the patches to extract from each image.
        random_patches: Whether to extract random patches from the input. If
            false, it will instead extract a set of standardized patches.
        shuffle: If true, it will shuffle the data in the dataset randomly.
            Disable if you want the output to always be deterministic.

    Returns:
        A dataset that produces input images and density maps.

    """
    # Deserialize it.
    feature_dataset = raw_dataset.map(
        _parse_example, num_parallel_calls=_NUM_THREADS
    )

    # Shuffle the data so we get different batches every time.
    if shuffle:
        feature_dataset = feature_dataset.shuffle(
            batch_size * num_prefetch_batches, reshuffle_each_iteration=True
        )

    # Batch and wrangle it.
    batched = feature_dataset.batch(batch_size)
    density_dataset = batched.map(
        functools.partial(
            _load_from_feature_dict, map_shape=map_shape, sigma=sigma
        ),
        num_parallel_calls=_NUM_THREADS,
    )
    # Extract patches.
    if random_patches:
        patched_dataset = density_dataset.map(
            lambda i, d: extract_random_patches(
                images=i, density_maps=d, patch_scale=patch_scale
            ),
            num_parallel_calls=_NUM_THREADS,
        )
    else:
        patched_dataset = density_dataset.interleave(
            lambda i, d: extract_standard_patches(
                images=i, density_maps=d, patch_scale=patch_scale
            ),
            cycle_length=_NUM_THREADS,
            num_parallel_calls=_NUM_THREADS,
        )
        # This effectively removed the batching, so we need to re-batch.
        patched_dataset = patched_dataset.batch(batch_size)
    # Compute total counts.
    patches_with_counts = patched_dataset.map(
        lambda i, d: _add_counts(
            images=i, density_maps=d, bucket_min_values=bucket_min_values
        ),
        num_parallel_calls=_NUM_THREADS,
    )

    # Prefetch the batches.
    return patches_with_counts.prefetch(num_prefetch_batches)
