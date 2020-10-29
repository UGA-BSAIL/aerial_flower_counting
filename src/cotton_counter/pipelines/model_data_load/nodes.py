"""
Nodes for the `model_data_load` pipeline.
"""


import tensorflow as tf


def combine_point_and_tag_datasets(
    *,
    point_dataset: tf.data.Dataset,
    tag_dataset_positive: tf.data.Dataset,
    tag_dataset_negative: tf.data.Dataset,
    tag_fraction: float,
    positive_repetitions: int,
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
        positive_repetitions: How many times to repeat the positive dataset
            in order to have the positive and negative annotations be better
            balanced.
        batch_size: The size of the batches in the datasets.

    Returns:
        A new combined dataset that randomly chooses elements from both inputs.

    """
    # Un-batch everything so that it gets mixed within batches.
    tag_dataset_positive = tag_dataset_positive.unbatch()
    tag_dataset_negative = tag_dataset_negative.unbatch()
    point_dataset = point_dataset.unbatch()

    # Combine the positive and negative datasets into one (balanced) dataset.
    positive_repeated = tag_dataset_positive.repeat(positive_repetitions)
    tag_dataset = tf.data.experimental.sample_from_datasets(
        [positive_repeated, tag_dataset_negative]
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
    # Re-batch once we've combined.
    return combined.batch(batch_size)
