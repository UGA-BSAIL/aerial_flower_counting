"""
Tests for the `dataset_io` module.
"""


import numpy as np
import pytest
import tensorflow as tf
import yaml
from pytest_snapshot.plugin import Snapshot

from src.cotton_counter.model.dataset_io import inputs_and_targets_from_dataset

from .data import IMAGE_SHAPE, TEST_DATASET_PATH


@pytest.mark.integration
@pytest.mark.slow
def test_point_dataset_integration(snapshot: Snapshot) -> None:
    """
    Verifies that we get consistent results when we try to load some example
    data.

    Args:
        snapshot: The fixture to use for snapshot testing.

    """
    # Arrange.
    # Load the raw data.
    raw_data = tf.data.TFRecordDataset([TEST_DATASET_PATH.as_posix()])

    # Act.
    # Process the data.
    patch_data = inputs_and_targets_from_dataset(
        raw_data,
        batch_size=16,
        num_prefetch_batches=10,
        image_shape=IMAGE_SHAPE,
        map_shape=(432, 576),
        sigma=3.0,
        patch_scale=0.125,
        random_patches=False,
        shuffle=False,
        include_counts=True,
    )

    # Assert.
    # Extract all the target data.
    counts = [t["count"] for _, t in patch_data]
    counts = np.stack(counts, axis=1)
    discrete_counts = [t["discrete_count"] for _, t in patch_data]
    discrete_counts = np.stack(discrete_counts, axis=1)

    # Serialize the result.
    all_targets = {
        "count": counts.tolist(),
        "discrete_count": discrete_counts.tolist(),
    }
    serial_targets = yaml.dump(all_targets, Dumper=yaml.Dumper)

    snapshot.assert_match(serial_targets, "dataset_targets.yaml")
