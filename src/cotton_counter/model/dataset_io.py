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
from .patches import (
    extract_random_annotated_patches,
    extract_standard_annotated_patches,
    extract_standard_patches,
)
from .records import Annotations

_POINTS_FEATURE_DESCRIPTION = {
    "image": tf.io.FixedLenFeature([1], tf.dtypes.string),
    "frame_numbers": tf.io.RaggedFeature(tf.dtypes.int64),
    "annotation_x": tf.io.RaggedFeature(tf.dtypes.float32),
    "annotation_y": tf.io.RaggedFeature(tf.dtypes.float32),
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
_INTERNAL_BATCH_SIZE_DIVISOR = 2
"""
The initial parts of the pipeline work with batches for computational
efficiency. However, using large batches here increases memory usage. The
internal batch size to use here is derived from the (output) batch size divided
by this constant.
"""


def _decode_jpegs(jpeg_batch: tf.Tensor) -> tf.Tensor:
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


def _parse_examples(
    serialized: tf.Tensor,
    feature_schema: Dict[str, Any] = _POINTS_FEATURE_DESCRIPTION,
) -> Dict[str, tf.Tensor]:
    """
    Deserializes a set of serialized examples.

    Args:
        serialized: The tensor containing a batch of serialized examples.
        feature_schema: The schema to use when extracting features.

    Returns:
        The parsed feature dictionary.

    """
    return tf.io.parse_example(serialized, feature_schema)


def _transform_to_patches(
    feature_dataset: tf.data.Dataset,
    *,
    image_shape: Vector2I,
    patch_scale: float,
    random_patches: bool,
) -> tf.data.Dataset:
    """
    Extracts image patches from a dataset of parsed feature dictionaries.

    Args:
        feature_dataset: The dataset containing parsed feature dictionaries.
            It expects these to already be divided into batches.
        image_shape: The shape of the input images, in the form
            (height, width).
        patch_scale: The scale factor to use for the patches.
        random_patches: If true, extract random patches. Otherwise, extract
            standardized patched.

    Returns:
        A dataset that is similar in structure to the input, but contains
        patches instead of full images. Also, all batching will have been
        removed.

    """

    def _process_with_annotations(
        feature_dict: Dict[str, tf.Tensor],
        patcher: Callable[..., tf.data.Dataset],
    ) -> tf.data.Dataset:
        """
        Helper function that manipulates the annotations from a feature
        dictionary into an `Annotations` object and extracts patches. Note
        that it expects the images to be still encoded in JPEG form and will
        automatically decode them.

        Args:
            feature_dict: The feature dictionary.
            patcher: The function to use for extracting patches.

        Returns:
            The extracted patches, in a new dataset.

        """
        x_values = feature_dict["annotation_x"]
        y_values = feature_dict["annotation_y"]
        annotations = Annotations(x_values=x_values, y_values=y_values,)

        # Decode the images.
        images = _decode_jpegs(feature_dict["image"])
        # Set static shapes for the images. We assume that each image has 3
        # channels.
        images = tf.map_fn(
            lambda i: tf.ensure_shape(i, tuple(image_shape) + (3,)), images
        )

        return patcher(
            images=images, annotations=annotations, patch_scale=patch_scale
        )

    # Choose the patch extraction function.
    if random_patches:
        patch_extractor = functools.partial(
            _process_with_annotations, patcher=extract_random_annotated_patches
        )
    else:
        patch_extractor = functools.partial(
            _process_with_annotations,
            patcher=extract_standard_annotated_patches,
        )

    return feature_dataset.interleave(
        patch_extractor,
        cycle_length=_NUM_THREADS,
        num_parallel_calls=_NUM_THREADS,
    )


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
    images = _decode_jpegs(feature_dict["image"])
    # Assume the images have three channels.
    images = tf.ensure_shape(images, (None, None, None, 3))

    discrete_counts = feature_dict["has_flower"]
    # Coerce into the form that Keras likes.
    discrete_counts = discrete_counts[:, 0]
    discrete_counts = tf.cast(discrete_counts, tf.int32)

    return {"image": images}, {"has_flower": discrete_counts}


def _binarize_counts(counts: tf.Tensor) -> tf.Tensor:
    """
    Binarizes the counts into positive and negative classes based on whether
    there is a flower or not.

    Args:
        counts: The tensor containing raw count values.

    Returns:
        An integer tensor containing the categorical counts.

    """
    return tf.where(tf.greater(counts, 0), tf.constant(1), tf.constant(0))


def _counts_from_feature_dict(
    *, feature_dict: Dict[str, tf.Tensor], include_counts: bool = True,
) -> Dict[str, tf.Tensor]:
    """
    Adds summed count values to a `Dataset`.

    Args:
        feature_dict: The deserialized feature dictionary for a single frame.
        include_counts: If true, include the raw counts in the targets.

    Returns:
        A dictionary mapping count and discrete count outputs to tensors.

    """
    # Find the number of annotations.
    x_values = feature_dict["annotation_x"]
    count = tf.size(x_values)

    # Discretize the counts.
    discrete_counts = _binarize_counts(count,)

    targets = dict(has_flower=discrete_counts)
    if include_counts:
        targets["count"] = count
    return targets


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


def _extract_from_points_feature_dict(
    feature_dataset: tf.data.Dataset,
    *,
    include_counts: bool = False,
    batch_size: int,
) -> tf.data.Dataset:
    """
    Extracts and wrangles relevant data from the parsed feature dictionaries. It
    will extract patches, and combine the data into batches.

    Args:
        feature_dataset: The input dataset containing parsed feature
            dictionaries. It expects the JPEG images to have already been
            decoded, and patches to have already been extracted.
        include_counts: If true, include the raw counts in the targets.
        batch_size: The size of the batches that we generate.

    Returns:
        A dataset containing a dictionary of inputs, and a dictionary of
        targets, divided into batches.

    """

    def add_counts(feature_dict: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        Stage 1: Adds "count" and "discrete_count" keys to the feature
        dictionary.

        Args:
            feature_dict: The input feature dictionary.

        Returns:
            The output feature dictionary.

        """
        count_features = _counts_from_feature_dict(
            feature_dict=feature_dict, include_counts=include_counts
        )

        feature_dict.update(count_features)
        return feature_dict

    features_with_counts = feature_dataset.map(
        add_counts, num_parallel_calls=_NUM_THREADS
    )

    def split_inputs_and_targets(
        feature_dict: Dict[str, tf.Tensor]
    ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        """
        Stage 2: Splits a feature dictionary into separate inputs and targets
        dictionaries with extraneous keys removed.

        Args:
            feature_dict: The feature dictionary to split.

        Returns:
            The two separate inputs and targets dictionaries.

        """
        inputs = dict(image=feature_dict["image"])
        targets = dict(has_flower=feature_dict["has_flower"],)
        if include_counts:
            targets["count"] = feature_dict["count"]

        return inputs, targets

    with_correct_schema = features_with_counts.map(
        split_inputs_and_targets, num_parallel_calls=_NUM_THREADS,
    )

    return with_correct_schema.batch(batch_size)


def load_point_dataset(
    raw_dataset: tf.data.Dataset, batch_size: int = 32
) -> tf.data.Dataset:
    """
    Loads a raw dataset containing point annotations. Will perform no further
    transformations.

    Args:
        raw_dataset: The raw dataset, containing serialized data.
        batch_size: The size of the batches that we generate.

    Returns:
        A dataset that produces the parsed data.

    """
    # Deserialize it.
    batched_raw_dataset = raw_dataset.batch(batch_size)
    return batched_raw_dataset.map(
        _parse_examples, num_parallel_calls=_NUM_THREADS
    )


def inputs_and_targets_from_dataset(
    raw_dataset: tf.data.Dataset,
    *,
    image_shape: Vector2I,
    batch_size: int = 32,
    num_prefetch_batches: int = 5,
    patch_scale: float = 1.0,
    random_patches: bool = True,
    include_counts: bool = False,
) -> tf.data.Dataset:
    """
    Deserializes raw data from a `Dataset` containing full images and point
    annotations, and coerces it into the form used by the model.

    Args:
        raw_dataset: The raw dataset, containing serialized data.
        image_shape: The shape of the input images, in the form
            (height, width).
        batch_size: The size of the batches that we generate.
        num_prefetch_batches: The number of batches to prefetch into memory.
            Increasing this can increase performance at the expense of memory
            usage.
        patch_scale: Scale of the patches to extract from each image.
        random_patches: Whether to extract random patches from the input. If
            false, it will instead extract a set of standardized patches.
        include_counts: If true, include the raw flower count for the patch
            in the targets under the "count" key.

    Returns:
        A dataset that produces input images and density maps.

    """
    feature_dataset = load_point_dataset(
        raw_dataset, batch_size=batch_size // _INTERNAL_BATCH_SIZE_DIVISOR
    )

    # Extract patches.
    patch_dataset = _transform_to_patches(
        feature_dataset,
        image_shape=image_shape,
        patch_scale=patch_scale,
        random_patches=random_patches,
    )
    # Re-batch and wrangle it.
    patches_with_counts = _extract_from_points_feature_dict(
        patch_dataset, include_counts=include_counts, batch_size=batch_size,
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

    Returns:
        A dataset that produces input images and a flag indicating whether
        the image contains at least one flower.

    """
    # Deserialize it.
    batched_raw_dataset = raw_dataset.batch(batch_size)
    feature_dataset = batched_raw_dataset.map(
        lambda x: _parse_examples(x, feature_schema=_TAG_FEATURE_DESCRIPTION),
        num_parallel_calls=_NUM_THREADS,
    )

    # Wrangle the format.
    discrete_count_dataset = feature_dataset.map(
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
        return dict(path=image_path, **element)

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
