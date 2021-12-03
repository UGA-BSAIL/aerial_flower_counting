"""
Nodes for the `model_data_load` pipeline.
"""


from typing import Sequence

import numpy as np
import tensorflow as tf
from loguru import logger

from ...model.hard_negatives import ThresholdType, filter_hard_negatives


class DatasetManager:
    """
    Manages datasets, including combining, balancing, and hard
    negative mining.
    """

    def __init__(
        self,
        *,
        point_dataset: tf.data.Dataset,
        tag_dataset_positive: tf.data.Dataset,
        tag_dataset_negative: tf.data.Dataset,
        num_positive_patches: int,
        num_negative_patches: int,
    ):
        """
        Combines a dataset containing point annotations and one containing tag
        annotations. Note that it will strip out all targets from the point
        dataset except for discrete counts.

        Args:
            point_dataset: The dataset containing point annotations.
            tag_dataset_positive: The dataset containing positive tag
                annotations.
            tag_dataset_negative: The dataset containing negative tag
                annotations.
            num_positive_patches: The number of positive examples in the tagged
                patch dataset.
            num_negative_patches: The number of negative examples in the tagged
                patch dataset.

        Returns:
            A new combined dataset that randomly chooses elements from both
            inputs.

        """
        # Strip extraneous targets from the point dataset.
        point_dataset_stripped = point_dataset.map(
            lambda i, t: (i, {"has_flower": t["has_flower"]})
        )

        # We un-batch everything, so that we can easily mix it across batches.
        self.__point_dataset = point_dataset_stripped.unbatch()
        self.__tag_dataset_positive = tag_dataset_positive.unbatch()
        self.__tag_dataset_negative = tag_dataset_negative.unbatch()

        self.__num_positive_patches = num_positive_patches
        self.__num_negative_patches = num_negative_patches
        self.__minority_length = min(
            self.__num_negative_patches, self.__num_positive_patches
        )
        logger.debug(
            "Smallest tag dataset has {} examples.", self.__minority_length
        )

        # Balanced versions of the tag datasets with associated sizes.
        self.__balanced_negatives = self.__tag_dataset_negative.take(
            self.__minority_length
        )
        self.__balanced_positives = self.__tag_dataset_positive.take(
            self.__minority_length
        )

    @staticmethod
    def __combine_datasets(
        datasets: Sequence[tf.data.Dataset], lengths: Sequence[int]
    ) -> tf.data.Dataset:
        """
        Combines multiple datasets into one by randomly shuffling them together.

        Args:
            datasets: The datasets to combine.
            lengths: The corresponding lengths of the datasets.

        Returns:
            The combined dataset.

        """
        # Compute sampling weights based on the lengths.
        total_length = sum(lengths)
        logger.debug("Shuffled datasets will have length of {}.", total_length)
        weights = [length / total_length for length in lengths]

        return tf.data.experimental.sample_from_datasets(
            datasets, weights=weights
        )

    def rebalance(self, *, model: tf.keras.Model) -> None:
        """
        Re-balances the dataset through hard negative (or positive) mining.
        After calling this, calling `get_combined()` will return the new
        balanced dataset.

        Args:
            model: The partially-trained model to use for finding hard
                negatives (or positives).

        """
        # Find the minority dataset.
        majority_dataset = self.__tag_dataset_negative
        # A zero output from the model means positive.
        threshold_type = ThresholdType.LOW_SCORE
        if self.__num_negative_patches < self.__num_positive_patches:
            majority_dataset = self.__tag_dataset_positive
            threshold_type = ThresholdType.HIGH_SCORE

        # Do the hard negative mining. (The same function works for hard
        # positives too with the proper threshold type.)
        reduced_majority_dataset = filter_hard_negatives(
            negative_patches=majority_dataset,
            threshold_type=threshold_type,
            model=model,
            num_to_keep=self.__minority_length,
        )

        if self.__num_negative_patches < self.__num_positive_patches:
            self.__balanced_positives = reduced_majority_dataset
        else:
            self.__balanced_negatives = reduced_majority_dataset

    def get_combined(
        self, *, tag_fraction: float, batch_size: int
    ) -> tf.data.Dataset:
        """
        Creates a combined dataset from all the inputs.

        Args:
            tag_fraction: The fraction of elements of the resulting dataset to
                draw from the tag dataset.
            batch_size: The size of the batches in the datasets.

        Returns:
            The combined dataset.

        """
        tag_dataset = self.__combine_datasets(
            (self.__tag_dataset_negative, self.__tag_dataset_positive),
            (self.__num_negative_patches, self.__num_positive_patches),
        )

        # Weights for random sampling.
        assert 0.0 <= tag_fraction <= 1.0, "tag_fraction must be in [0.0, 1.0]"
        sample_weights = [tag_fraction, 1.0 - tag_fraction]

        combined = tf.data.experimental.sample_from_datasets(
            [tag_dataset, self.__point_dataset], weights=sample_weights
        )

        # Shuffle the data.
        combined = combined.shuffle(
            batch_size * 2, reshuffle_each_iteration=True
        )
        # Re-batch once we've combined.
        return add_dummy_targets(
            combined.batch(batch_size), batch_size=batch_size
        )


def add_dummy_targets(
    dataset: tf.data.Dataset, *, batch_size: int
) -> tf.data.Dataset:
    """
    Modifies a dataset to add a dummy target for the scale consistency and
    combined BCE losses. We don't actually need a target to compute these
    losses, but Keras requires that we have one.

    Args:
        dataset: The dataset to modify.
        batch_size: The batch size to use.

    Returns:
        The modified dataset.

    """
    zero = tf.zeros((batch_size,), dtype=tf.float32)

    return dataset.map(
        lambda i, t: (
            i,
            dict(combined_bce_loss=zero, scale_consistency_loss=zero, **t),
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
    # Compute the number of examples in the point dataset.
    total_num_patches = num_positive_patches + num_negative_patches
    total_num_points = total_num_patches / tag_fraction - total_num_patches

    # Compute the positive fraction in the combined dataset.
    num_positive_points = total_num_points * point_positive_fraction
    total_num_positive = int(num_positive_points) + num_positive_patches
    total_num_examples = (
        total_num_points + num_positive_patches + num_negative_patches
    )
    logger.info(
        "Dataset positive example fraction: {}.",
        total_num_positive / total_num_examples,
    )
    total_num_negative = int(total_num_examples) - total_num_positive

    return np.log(total_num_positive / total_num_negative)
