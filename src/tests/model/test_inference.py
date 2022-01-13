"""
Tests for the `inference` module.
"""


import cv2
import numpy as np
import tensorflow as tf

from src.cotton_counter.model import inference

from .data import TEST_IMAGE_PATH


def test_recombine_patches() -> None:
    """
    Tests that the `recombine_patches` function works.

    """
    # Arrange.
    # Load an image for testing.
    test_image = cv2.imread(TEST_IMAGE_PATH.as_posix(), cv2.IMREAD_GRAYSCALE)
    test_image_4d = tf.constant(
        tf.expand_dims(tf.expand_dims(test_image, 0), -1)
    )

    # Extract patches.
    kernel_size = [1, 100, 100, 1]
    flat_patches = tf.image.extract_patches(
        images=test_image_4d,
        sizes=kernel_size,
        strides=kernel_size,
        rates=[1, 1, 1, 1],
        padding="SAME",
    )
    # Expand the patches into their own batch.
    got_patches = tf.reshape(flat_patches, (-1, 100, 100, 1),)

    # Act.
    recombined_image = inference.recombine_patches(
        got_patches, images=test_image_4d, kernel_size=kernel_size
    )

    # Assert.
    # It should be identical to the input.
    np.testing.assert_array_equal(
        recombined_image.numpy(), test_image_4d.numpy()
    )
