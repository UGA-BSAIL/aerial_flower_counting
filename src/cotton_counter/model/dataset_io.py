"""
Extracts the information that we need to train the model from a raw
Tensorflow `Dataset`.
"""

import functools
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from loguru import logger

from ..type_helpers import Vector2I
from .density_maps import make_density_maps
from .patches import (
    extract_random_patches,
    extract_standard_patches,
    extract_standard_patches_from_dataset,
)
from .records import Annotations

_POINTS_FEATURE_DESCRIPTION = {
    "image": tf.io.FixedLenFeature([1], tf.dtypes.string),
    "frame_numbers": tf.io.VarLenFeature(tf.dtypes.int64),
    "annotation_x": tf.io.VarLenFeature(tf.dtypes.float32),
    "annotation_y": tf.io.VarLenFeature(tf.dtypes.float32),
}
"""
Descriptions of the features found in datasets containing point annotations.
"""

_TAG_FEATURE_DESCRIPTION = {
    "image": tf.io.FixedLenFeature([1], tf.dtypes.string),
    "frame_number": tf.io.FixedLenFeature([1], tf.dtypes.int64),
    "has_flower": tf.io.FixedLenFeature([1], tf.dtypes.int64),
}
"""
Description of the features found in datasets containing tag annotations.
"""

_NUM_THREADS = cpu_count()
"""
Number of threads to use for multi-threaded operations.
"""


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


def _parse_example(
    serialized: tf.Tensor,
    feature_schema: Dict[str, Any] = _POINTS_FEATURE_DESCRIPTION,
) -> Dict[str, tf.Tensor]:
    """
    Deserializes a set of serialized examples.

    Args:
        serialized: The tensor containing a serialized example.
        feature_schema: The schema to use when extracting features.

    Returns:
        The parsed feature dictionary.

    """
    return tf.io.parse_single_example(serialized, feature_schema)


def _load_from_points_feature_dict(
    feature_dict: Dict[str, tf.Tensor], *, map_shape: Vector2I, sigma: float
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Wrangles the data from a parsed feature dictionary for points annotations
    into images and density maps.

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


def _load_from_tag_feature_dict(
    feature_dict: Dict[str, tf.Tensor]
) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    """
    Wrangles the data from a parsed feature dictionary for tag annotations
    into images and discrete counts.

    Args:
        feature_dict: The parsed feature dictionary.

    Returns:
        The images from the batch as well as the generated discrete count
        vector. This is produced as two feature dictionaries, containing
        model inputs and targets.

    """
    # Decode the JPEG images.
    images = _decode_jpeg(feature_dict["image"])
    # Assume the images have three channels.
    images = tf.ensure_shape(images, (None, None, None, 3))

    # For discrete counts, class 0 is actually the positive class, so we just
    # have to flip the tag annotations around.
    discrete_counts = (
        tf.constant(1, dtype=tf.int64) - feature_dict["has_flower"]
    )
    # Coerce into the form that Keras likes.
    discrete_counts = discrete_counts[:, 0]
    discrete_counts = tf.cast(discrete_counts, tf.int32)

    return {"image": images}, {"discrete_count": discrete_counts}


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
    include_counts: bool = True,
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
        include_counts: If true, include the raw counts in the targets.

    Returns:
        A dictionary mapping model inputs to tensors and a dictionary mapping
        model outputs to tensors.

    """
    counts = tf.reduce_sum(density_maps, axis=[1, 2, 3])

    # Discretize the counts.
    discrete_counts = _discretize_counts(
        counts, bucket_min_values=bucket_min_values
    )

    targets = dict(discrete_count=discrete_counts)
    if include_counts:
        targets["count"] = counts
    return dict(image=images), targets


def _extract_patches(
    image_dataset: tf.data.Dataset,
    *,
    patch_scale: float,
    random_patches: bool,
    batch_size: int,
    image_shape: Vector2I,
    map_shape: Vector2I,
) -> tf.data.Dataset:
    """
    Extracts patches from input images and density maps.

    Args:
        image_dataset: An input dataset where each element contains a raw image
            and a corresponding density map.
        patch_scale: Scale of the patches to extract from each image.
        random_patches: Whether to extract random patches from the input. If
            false, it will instead extract a set of standardized patches.
        batch_size: The size of the batches in the input dataset.
        image_shape: The shape of the input images, in the form
            (height, width).
        map_shape: The shape of the output density maps to generate, in the
            form (height, width).

    Returns:
        A new dataset containing the patches from the raw images and density
        maps.

    """
    # Compute full static shapes for each batch. (We can't specify the batch
    # size statically because the last one might be smaller.)
    image_shape = tuple(image_shape)
    map_shape = tuple(map_shape)
    full_image_shape = (None,) + image_shape + (3,)
    full_density_shape = (None,) + map_shape + (1,)
    images_with_shapes = image_dataset.map(
        lambda i, d: (
            tf.ensure_shape(i, full_image_shape),
            tf.ensure_shape(d, full_density_shape),
        )
    )

    if random_patches:
        patched_dataset = images_with_shapes.map(
            lambda i, d: extract_random_patches(
                images=i, density_maps=d, patch_scale=patch_scale
            ),
            num_parallel_calls=_NUM_THREADS,
        )
    else:
        patched_dataset = images_with_shapes.interleave(
            lambda i, d: extract_standard_patches_from_dataset(
                images=i, density_maps=d, patch_scale=patch_scale
            ),
            cycle_length=_NUM_THREADS,
            num_parallel_calls=_NUM_THREADS,
        )
        # This effectively removed the batching, so we need to re-batch.
        patched_dataset = patched_dataset.batch(batch_size)

    return patched_dataset


def _repeat_patches(
    dataset: tf.data.Dataset, *, patch_scale: float
) -> tf.data.Dataset:
    """
    Repeats a dataset to account for the increased number of items due to
    patching. The intuition behind this is that patches effectively increase
    the size of the dataset by a certain amount, so what we're calling an
    "epoch" should be longer to account for the various patch permutations.

    Args:
        dataset: The dataset that we want to repeat.
        patch_scale: The patch scale that we used when extracting patches.

    Returns:
        The repeated dataset.

    """
    # Figure out how many times to repeat.
    num_patches_1d = int(1.0 / patch_scale)
    num_patches_2d = num_patches_1d ** 2
    logger.debug(
        "Repeating dataset {} times to account for patches.", num_patches_2d
    )

    return dataset.repeat(num_patches_2d)


def _write_to_file(
    data: tf.Tensor,
    *,
    index: tf.Tensor,
    save_dir: Path,
    extension: str = ".jpg",
) -> tf.Tensor:
    """
    Saves data to files on the disk without otherwise changing it.

    Args:
        data: The raw data to save.
        save_dir: The directory to write files to.
        index: Unique index for this item in the dataset. It will be used to
            generate the file name.
        extension: The file extension to use for saved data.

    Returns:
        The path to the file where these data are stored.

    """
    tf.assert_rank(index, 0)

    # Generate the path to the file.
    string_index = tf.strings.as_string(index)
    file_name = tf.strings.join(["element_", string_index, extension])
    file_path = tf.strings.join([save_dir.as_posix(), "/", file_name])

    # Save the file.
    tf.io.write_file(file_path, data)

    return file_path


def inputs_and_targets_from_dataset(
    raw_dataset: tf.data.Dataset,
    *,
    image_shape: Vector2I,
    map_shape: Vector2I,
    sigma: float,
    bucket_min_values: List[float],
    batch_size: int = 32,
    num_prefetch_batches: int = 5,
    patch_scale: float = 1.0,
    random_patches: bool = True,
    shuffle: bool = True,
    include_counts: bool = False,
) -> tf.data.Dataset:
    """
    Deserializes raw data from a `Dataset` containing full images and point
    annotations, and coerces it into the form used by the model.

    Args:
        raw_dataset: The raw dataset, containing serialized data.
        image_shape: The shape of the input images, in the form
            (height, width).
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
        include_counts: If true, include the raw flower count for the patch
            in the targets under the "count" key.

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
            _load_from_points_feature_dict, map_shape=map_shape, sigma=sigma
        ),
        num_parallel_calls=_NUM_THREADS,
    )
    # Extract patches.
    patched_dataset = _extract_patches(
        density_dataset,
        patch_scale=patch_scale,
        random_patches=random_patches,
        batch_size=batch_size,
        image_shape=image_shape,
        map_shape=map_shape,
    )
    # Compute total counts.
    patches_with_counts = patched_dataset.map(
        lambda i, d: _add_counts(
            images=i,
            density_maps=d,
            bucket_min_values=bucket_min_values,
            include_counts=include_counts,
        ),
        num_parallel_calls=_NUM_THREADS,
    )

    if random_patches:
        # For random patches, repeat the dataset to account for the large
        # number of possible patches.
        patches_with_counts = _repeat_patches(
            patches_with_counts, patch_scale=patch_scale
        )
    # Prefetch the batches.
    return patches_with_counts.prefetch(num_prefetch_batches)


def inputs_and_targets_from_patch_dataset(
    raw_dataset: tf.data.Dataset,
    *,
    batch_size: int = 32,
    num_prefetch_batches: int = 5,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """
    Deserializes raw data from a `Dataset` containing patches and tag
    annotations, and coerces it into the form used by the model.

    Args:
        raw_dataset: The raw dataset, containing serialized data.
        batch_size: The size of the batches that we generate.
        num_prefetch_batches: The number of batches to prefetch into memory.
            Increasing this can increase performance at the expense of memory
            usage.
        shuffle: If true, it will shuffle the data in the dataset randomly.
            Disable if you want the output to always be deterministic.

    Returns:
        A dataset that produces input images and a flag indicating whether
        the image contains at least one flower.

    """
    # Deserialize it.
    feature_dataset = raw_dataset.map(
        lambda x: _parse_example(x, feature_schema=_TAG_FEATURE_DESCRIPTION),
        num_parallel_calls=_NUM_THREADS,
    )

    # Shuffle the data so we get different batches every time.
    if shuffle:
        feature_dataset = feature_dataset.shuffle(
            batch_size * num_prefetch_batches, reshuffle_each_iteration=True
        )

    # Batch and wrangle it.
    batched = feature_dataset.batch(batch_size)
    discrete_count_dataset = batched.map(
        functools.partial(_load_from_tag_feature_dict),
        num_parallel_calls=_NUM_THREADS,
    )

    # Prefetch the batches.
    return discrete_count_dataset.prefetch(num_prefetch_batches)


def save_images_to_disk(
    images: tf.data.Dataset, *, save_dir: Path
) -> tf.data.Dataset:
    """
    Saves images contained in a `Dataset` to files on the disk.

    Args:
        images: The `Dataset` containing images. The images should be in raw
            form, and it expects each element to be a dictionary, where the
            image data is under the "image" key.
        save_dir:
            The directory in which to create the image files.

    Returns:
        A new `Dataset` that contains dictionaries with two keys. The
        "image" key refers to the original images. The "path" key refers to
        the path that this image is saved at.

    """
    # We need unique indices for each image.
    numbered_images = images.enumerate()

    def encode_and_write(
        element: Dict[str, tf.Tensor], **kwargs: Any
    ) -> Dict[str, tf.Tensor]:
        # Encode the image.
        image = element["image"]
        encoded = tf.io.encode_jpeg(image)

        # Save the image.
        image_path = _write_to_file(encoded, **kwargs)
        return dict(image=image, path=image_path)

    return numbered_images.map(
        lambda i, e: encode_and_write(
            e, index=i, save_dir=save_dir, extension=".jpg"
        )
    )


def inputs_from_generator(
    image_source: Callable[[], Iterable[np.ndarray]],
    *,
    image_shape: Vector2I,
    batch_size: int = 32,
    num_prefetch_batches: int = 5,
    patch_scale: float = 1.0,
    patch_stride: Optional[float] = None,
    extract_jpegs: bool = False,
) -> tf.data.Dataset:
    """
    Produces a model input dataset from a dataset containing raw images.

    Args:
        image_source: A generator that yields the raw RGB images that we want
           to use as input.
        image_shape: The shape of the input images, in the form
            (height, width).
        batch_size: The size of the batches that we generate.
        num_prefetch_batches: The number of batches to prefetch into memory.
            Increasing this can increase performance at the expense of memory
            usage.
        patch_scale: Scale of the patches to extract from each image.
        patch_stride: The stride to use for extracted patches. If not
            specified, it will be the same as the patch scale, which produces
            patches that don't overlap.
        extract_jpegs: If true, it expects the input images to be JPEGs, and
            will extract them as part of the loading procedure. Otherwise, it
            expects them to be raw images.

    Returns:
        A dataset containing the model input.

    """
    raw_images = tf.data.Dataset.from_generator(image_source, tf.string)
    # Decode JPEGS if necessary.
    if extract_jpegs:
        raw_images = raw_images.map(tf.image.decode_jpeg)

    # Set static shapes for the images. We assume that each image has 3
    # channels.
    raw_images = raw_images.map(
        lambda i: tf.ensure_shape(i, tuple(image_shape) + (3,)),
    )

    # Combine images into batches.
    raw_images_batched = raw_images.batch(batch_size)

    # Extract standard patches from each image.
    patches = raw_images_batched.flat_map(
        lambda i: tf.data.Dataset.from_tensor_slices(
            extract_standard_patches(
                i, patch_scale=patch_scale, patch_stride=patch_stride
            )
        ),
    )
    # Patch extraction removed the batching, so we need to re-batch.
    patches = patches.batch(batch_size)

    # Impose the proper schema.
    def transform_schema(image: tf.Tensor) -> Dict[str, tf.Tensor]:
        return {"image": image}

    patches = patches.map(transform_schema)

    # Prefetch the batches.
    return patches.prefetch(num_prefetch_batches)
