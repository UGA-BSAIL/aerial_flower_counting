"""
Utilities for dealing with TFRecords data.
"""


from typing import Iterable, Union

import numpy as np
import tensorflow as tf


def bytes_feature(
    feature_bytes: Union[Iterable[np.uint8], bytes]
) -> tf.train.Feature:
    """
    Converts binary data to a Tensorflow feature.

    Args:
        feature_bytes: The input binary data.

    Returns:
        The resulting feature.

    """
    if isinstance(feature_bytes, np.ndarray):
        # Convert to Python bytes.
        feature_bytes = feature_bytes.tobytes()

    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[feature_bytes])
    )


def float_feature(feature_floats: Iterable[float]):
    """
    Converts float data to a Tensorflow feature.

    Args:
        feature_floats: The input float data.

    Returns:
        The resulting feature.

    """
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=list(feature_floats))
    )


def int_feature(feature_ints: Iterable[int]):
    """
    Converts integer data to a Tensorflow feature.

    Args:
        feature_ints: The input float data.

    Returns:
        The resulting feature.

    """
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(feature_ints))
    )
