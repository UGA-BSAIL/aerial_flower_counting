"""
Extracts the information that we need to train the model from a raw
Tensorflow `Dataset`.
"""

from functools import partial
from multiprocessing import cpu_count
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from loguru import logger

from ..type_helpers import Vector2I
from .density_maps import make_density_map
from .patches import extract_random_patch, extract_standard_patches
from .records import Annotations
from .resampling import balance_distribution

_FEATURE_DESCRIPTION = {
    "image": tf.io.FixedLenFeature([1], tf.dtypes.string),
    "frame_numbers": tf.io.VarLenFeature(tf.dtypes.int64),
    "annotation_x": tf.io.VarLenFeature(tf.dtypes.float32),
    "annotation_y": tf.io.VarLenFeature(tf.dtypes.float32),
}
"""
Descriptions of the features found in our dataset.
"""


NUM_THREADS = cpu_count()


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
        lambda j: tf.io.decode_jpeg(j[0]), jpeg_batch, dtype=tf.dtypes.uint8,
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


def _transform_to_patches(
    feature_dataset: tf.data.Dataset,
    *,
    patch_scale: float,
    random_patches: bool,
) -> tf.data.Dataset:
    """
    Extracts image patches from a dataset of parsed feature dictionaries.

    Args:
        feature_dataset: The dataset containing parsed feature dictionaries.
        patch_scale: The scale factor to use for the patches.
        random_patches: If true, extract random patches. Otherwise, extract
            standardized patched.

    Returns:
        A dataset that is identical in structure to the input, but contains
        patches instead of full images.

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
        # It is valid to make these dense since they only contain data for
        # one example.
        x_values = tf.sparse.to_dense(feature_dict["annotation_x"])
        y_values = tf.sparse.to_dense(feature_dict["annotation_y"])
        frame_numbers = tf.sparse.to_dense(feature_dict["frame_numbers"])

        annotations = Annotations(
            x_values=x_values, y_values=y_values, frame_numbers=frame_numbers,
        )
        # Decode the images.
        image = tf.io.decode_jpeg(feature_dict["image"][0])

        return patcher(
            image=image, annotations=annotations, patch_scale=patch_scale
        )

    # Choose the patch extraction function.
    if random_patches:
        patch_extractor = partial(
            _process_with_annotations, patcher=extract_random_patch
        )
    else:
        patch_extractor = partial(
            _process_with_annotations, patcher=extract_standard_patches
        )

    return feature_dataset.interleave(
        patch_extractor, num_parallel_calls=NUM_THREADS
    )


def _density_maps_from_feature_dict(
    feature_dict: Dict[str, tf.Tensor], *, map_shape: Vector2I, sigma: float
) -> tf.Tensor:
    """
    Creates density maps from a parsed feature dictionary.

    Args:
        feature_dict: The parsed feature dictionary.
        map_shape: The shape of the output density maps to generate, in the
            form (height, width). The samples dimension will be inferred.
        sigma:
            The standard deviation in pixels to use for the applied gaussian
            filter.

    Returns:
        The corresponding generated density maps.

    """
    x_values = feature_dict["annotation_x"]
    y_values = feature_dict["annotation_y"]

    # Create density maps.
    density_map = make_density_map(
        x_values, y_values, map_shape=map_shape, sigma=sigma
    )

    return density_map


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
    # Cast to floats for easy comparison.
    counts = tf.cast(counts, tf.float32)
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


def _counts_from_feature_dict(
    *, feature_dict: Dict[str, tf.Tensor], bucket_min_values: List[float],
) -> Dict[str, tf.Tensor]:
    """
    Extracts total count values from a `Dataset`.

    Args:
        feature_dict: The deserialized feature dictionary for a single frame.
        bucket_min_values: A list of the minimum count values that will go in
            each discrete count bucket. Note that the highest bucket will
            contain anything that falls between the largest minimum value and
            infinity.

    Returns:
        A dictionary mapping count and discrete count outputs to tensors.

    """
    # Find the number of annotations.
    frame_numbers = feature_dict["frame_numbers"]
    count = tf.size(frame_numbers)

    # Discretize the count.
    discrete_count = _discretize_counts(
        count, bucket_min_values=bucket_min_values
    )

    return dict(count=count, discrete_count=discrete_count)


def _balance_dataset(
    features_with_counts: tf.data.Dataset,
    *,
    dataset_size: int,
    bucket_min_values: List[float],
) -> tf.data.Dataset:
    """
    Re-samples the dataset such that it has close to equal numbers of examples
    from each density category.

    Args:
        features_with_counts: The dataset containing deserialized feature
            dictionaries with added keys for the corresponding count data.
        dataset_size: The size of the total dataset that we want to use. This
            should ideally be >= the total number of training examples we have.
        bucket_min_values: A list of the minimum count values that will go in
            each discrete count bucket. Note that the highest bucket will
            contain anything that falls between the largest minimum value and
            infinity.

    Returns:
        A dataset containing tuples of the original feature dictionaries and
        computed count and discrete count values.

    """
    # The basic method here is rather simple: We repeat the input dataset
    # infinitely and simply use rejection resampling until we reach our
    # desired dataset size.
    infinite_features = features_with_counts.repeat()

    # Create the rejection resampling transformation.
    resampled_features = balance_distribution(
        infinite_features,
        classify=lambda f: f["discrete_count"],
        num_classes=len(bucket_min_values),
    )

    return resampled_features.take(dataset_size)


def _extract_from_feature_dict(
    feature_dataset: tf.data.Dataset,
    *,
    map_shape: Vector2I,
    sigma: float,
    bucket_min_values: List[float],
    batch_size: int,
    balanced_size: Optional[int],
) -> tf.data.Dataset:
    """
    Extracts and wrangles relevant data from the parsed feature dictionaries. It
    will create density maps, and combine the data into batches.

    Args:
        feature_dataset: The input dataset containing parsed feature
            dictionaries. It expects the JPEG images to have already been
            decoded.
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
        balanced_size: If not None, it will try to have a balanced
            distribution of density classes in the output dataset. This number
            indicates the total number of examples we want in the balanced
            dataset.

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
            feature_dict=feature_dict, bucket_min_values=bucket_min_values
        )

        feature_dict.update(count_features)
        return feature_dict

    features_with_counts = feature_dataset.map(
        add_counts, num_parallel_calls=NUM_THREADS
    )

    # Stage 2: Balance the dataset if necessary.
    if balanced_size is not None:
        features_with_counts = _balance_dataset(
            features_with_counts,
            dataset_size=balanced_size,
            bucket_min_values=bucket_min_values,
        )

    def add_density_maps(
        feature_dict: Dict[str, tf.Tensor]
    ) -> Dict[str, tf.Tensor]:
        """
        Stage 3: Adds a "density_map" key to the feature dictionary.

        Args:
            feature_dict: The input feature dictionary.

        Returns:
            The output feature dictionary.

        """
        density_map = _density_maps_from_feature_dict(
            feature_dict=feature_dict, map_shape=map_shape, sigma=sigma
        )
        return dict(density_map=density_map, **feature_dict)

    features_counts_density = features_with_counts.map(
        add_density_maps, num_parallel_calls=NUM_THREADS
    )

    def split_inputs_and_targets(
        feature_dict: Dict[str, tf.Tensor]
    ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        """
        Stage 4: Splits a feature dictionary into separate inputs and targets
        dictionaries with extraneous keys removed.

        Args:
            feature_dict: The feature dictionary to split.

        Returns:
            The two separate inputs and targets dictionaries.

        """
        inputs = dict(image=feature_dict["image"])
        targets = dict(
            density_map=feature_dict["density_map"],
            count=feature_dict["count"],
            discrete_count=feature_dict["discrete_count"],
        )

        return inputs, targets

    with_correct_schema = features_counts_density.map(
        split_inputs_and_targets, num_parallel_calls=NUM_THREADS,
    )

    return with_correct_schema.batch(batch_size)


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
    balanced_size: Optional[int] = None,
) -> tf.data.Dataset:
    """
    Deserializes raw data from a `Dataset`, and coerces it into the form
    used by the model.

    Args:
        raw_dataset: The raw dataset, containing serialized data.
        map_shape: The shape of the output density maps to generate, in the
            form (height, width). Note that this will be adjusted
            automatically to take patching into account.
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
        balanced_size: If not None, it will try to have a balanced
            distribution of density classes in the output dataset. This number
            indicates the total number of examples we want in the balanced
            dataset.

    Returns:
        A dataset that produces input images and density maps.

    """
    # Adjust the map shape for the patch size.
    map_shape_float = np.array(map_shape) * patch_scale
    map_shape = tuple(map_shape_float.astype(np.int))
    logger.debug("Using patched map shape of {}.", map_shape)

    # Deserialize it.
    feature_dataset = raw_dataset.map(
        _parse_example, num_parallel_calls=NUM_THREADS
    )

    # Shuffle the data so we get different batches every time.
    if shuffle:
        feature_dataset = feature_dataset.shuffle(
            batch_size * num_prefetch_batches, reshuffle_each_iteration=True
        )

    # Extract patches.
    patch_dataset = _transform_to_patches(
        feature_dataset, patch_scale=patch_scale, random_patches=random_patches
    )
    # Batch and wrangle it.
    batched = _extract_from_feature_dict(
        patch_dataset,
        map_shape=map_shape,
        sigma=sigma,
        bucket_min_values=bucket_min_values,
        batch_size=batch_size,
        balanced_size=balanced_size,
    )

    # Prefetch the batches.
    return batched.prefetch(num_prefetch_batches)
