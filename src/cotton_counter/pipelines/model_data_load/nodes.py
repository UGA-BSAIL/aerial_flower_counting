"""
Nodes for the `model_data_load` pipeline.
"""


import numpy as np
import tensorflow as tf
from loguru import logger


def _make_balanced_tag_dataset(
    *,
    tag_dataset_positive: tf.data.Dataset,
    tag_dataset_negative: tf.data.Dataset,
    num_positive_patches: int,
    num_negative_patches: int,
) -> tf.data.Dataset:
    """
    Creates a single tagged patch example dataset that is balanced between
    the positive and negative classes. It does this by oversampling the
    minority class.

    Args:
        tag_dataset_positive: The dataset of positive examples.
        tag_dataset_negative: The dataset of negative examples.
        num_positive_patches: The number of examples in the positive dataset.
        num_negative_patches: The number of examples in the negative dataset.

    Returns:
        The combined, balanced dataset.

    """
    # Find the minority dataset.
    minority_dataset = tag_dataset_positive
    majority_dataset = tag_dataset_negative
    if num_negative_patches < num_positive_patches:
        minority_dataset = tag_dataset_negative
        majority_dataset = tag_dataset_positive

    # Repeat and cut the minority dataset so it is the same length as the
    # majority.
    majority_size = max(num_positive_patches, num_negative_patches)
    minority_dataset = minority_dataset.repeat()
    minority_dataset = minority_dataset.take(majority_size)

    # Combine by random sampling.
    return tf.data.experimental.sample_from_datasets(
        [minority_dataset, majority_dataset]
    )


def combine_point_and_tag_datasets(
    *,
    point_dataset: tf.data.Dataset,
    tag_dataset_positive: tf.data.Dataset,
    tag_dataset_negative: tf.data.Dataset,
    tag_fraction: float,
    num_positive_patches: int,
    num_negative_patches: int,
    batch_size: int,
) -> tf.data.Dataset:
    """
    Combines a dataset containing point annotations and one containing tag
    annotations. Note that it will strip out all targets from the point
    dataset except for discrete counts.

    Args:
        point_dataset: The dataset containing point annotations.
        tag_dataset_positive: The dataset containing positive tag annotations.
        tag_dataset_negative: The dataset containing negative tag annotations.
        tag_fraction: The fraction of elements of the resulting dataset to
            draw from the tag dataset.
        num_positive_patches: The number of positive examples in the tagged
            patch dataset.
        num_negative_patches: The number of negative examples in the tagged
            patch dataset.
        batch_size: The size of the batches in the datasets.

    Returns:
        A new combined dataset that randomly chooses elements from both inputs.

    """
    # Un-batch everything so that it gets mixed within batches.
    tag_dataset_positive = tag_dataset_positive.unbatch()
    tag_dataset_negative = tag_dataset_negative.unbatch()
    point_dataset = point_dataset.unbatch()

    # Combine the positive and negative datasets into one (balanced) dataset.
    tag_dataset = _make_balanced_tag_dataset(
        tag_dataset_positive=tag_dataset_positive,
        tag_dataset_negative=tag_dataset_negative,
        num_positive_patches=num_positive_patches,
        num_negative_patches=num_negative_patches,
    )

    # Strip extraneous targets from the point dataset.
    point_dataset_stripped = point_dataset.map(
        lambda i, t: (i, {"discrete_count": t["discrete_count"]})
    )

    # Weights for random sampling.
    assert 0.0 <= tag_fraction <= 1.0, "tag_fraction must be in [0.0, 1.0]"
    sample_weights = [tag_fraction, 1.0 - tag_fraction]

    combined = tf.data.experimental.sample_from_datasets(
        [tag_dataset, point_dataset_stripped], weights=sample_weights
    )

    # Shuffle the data.
    combined = combined.shuffle(batch_size * 2, reshuffle_each_iteration=True)

    # Re-batch once we've combined.
    return combined.batch(batch_size)


def add_sub_patch_target(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Modifies a dataset to add a target for the sub-patch classes, which is
    needed by Keras.

    Args:
        dataset: The dataset to add the sub-patch classes to.

    Returns:
        The modified dataset.

    """
    return dataset.map(
        lambda i, t: (
            i,
            dict(discrete_sub_patch_count=t["discrete_count"], **t),
        )
    )


def calculate_output_bias(
    *,
    point_positive_fraction: float,
    tag_fraction: float,
    num_positive_patches: int,
    num_negative_patches: int,
) -> float:
    """
    Calculates an initial bias value for the model output so that it starts
    by predicting roughly the correct distribution.

    Args:
        point_positive_fraction: The fraction of positive examples in the
            point dataset.
        tag_fraction: The fraction of elements of the combined dataset that
            are drawn from the tag dataset.
        num_positive_patches: The number of positive examples in the tagged
            patch dataset.
        num_negative_patches: The number of negative examples in the tagged
            patch dataset.

    Returns:
        The calculated initial bias.

    """
    # The oversampling algorithm is going to repeat examples from the
    # minority class so that there are the same number as the majority class.
    num_majority_examples = max(num_positive_patches, num_negative_patches)
    num_balanced_tag_examples = num_majority_examples * 2

    # Compute the number of examples in the point dataset.
    total_num_patches = num_positive_patches + num_negative_patches
    total_num_points = total_num_patches / tag_fraction - total_num_patches

    # Compute the positive fraction in the combined dataset.
    num_positive_points = total_num_points * point_positive_fraction
    total_num_positive = int(num_positive_points) + num_majority_examples
    total_num_examples = total_num_points + num_balanced_tag_examples
    logger.info(
        "Dataset positive example fraction: {}.",
        total_num_positive / total_num_examples,
    )
    total_num_negative = int(total_num_examples) - total_num_positive

    return np.log(total_num_positive / total_num_negative)
