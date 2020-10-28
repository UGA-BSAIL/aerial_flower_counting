"""
Nodes for the `model_data_load` pipeline.
"""


import tensorflow as tf


def combine_point_and_tag_datasets(
    *,
    point_dataset: tf.data.Dataset,
    tag_dataset: tf.data.Dataset,
    tag_fraction: float
) -> tf.data.Dataset:
    """
    Combines a dataset containing point annotations and one containing tag
    annotations. Note that it will strip out all targets from the point
    dataset except for discrete counts.

    Args:
        point_dataset: The dataset containing point annotations.
        tag_dataset: The dataset containing tag annotations.
        tag_fraction: The fraction of elements of the resulting dataset to
            draw from the tag dataset.

    Returns:
        A new combined dataset that randomly chooses elements from both inputs.

    """
    # Strip extraneous targets from the point dataset.
    point_dataset_stripped = point_dataset.map(
        lambda i, t: (i, {"discrete_count": t["discrete_count"]})
    )

    # Weights for random sampling.
    assert 0.0 <= tag_fraction <= 1.0, "tag_fraction must be in [0.0, 1.0]"
    sample_weights = [tag_fraction, 1.0 - tag_fraction]

    return tf.data.experimental.sample_from_datasets(
        [tag_dataset, point_dataset_stripped], weights=sample_weights
    )
