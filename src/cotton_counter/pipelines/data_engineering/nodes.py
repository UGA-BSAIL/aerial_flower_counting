"""
Contains node definitions for the `data_engineering` pipeline.
"""


from typing import Iterable, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from pycvat.dataset import CvatHandle


def make_splits(
    local_annotations: pd.DataFrame,
    train_fraction: float = 0.9,
    test_fraction: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates the train, test and validation splits for the data.

    Args:
        local_annotations: The `DataFrame` containing cleaned annotation data.
        train_fraction: The fraction of the data to use for training.
        test_fraction: The fraction of the data to use for testing.

    Returns:
        The train, test, and validation data as separate `DataFrame`s.

    """
    # We split on frames instead of points so we don't have data from the
    # same frame spread across multiple splits.
    frame_nums = local_annotations["frame_num"].unique()
    num_frames = len(frame_nums)
    logger.debug("Splitting data from {} frames.", num_frames)

    valid_fraction = 1.0 - train_fraction - test_fraction
    assert valid_fraction >= 0.0, "Train and test fraction should not be >1.0."
    train_split_size = int(num_frames * train_fraction)
    test_split_size = int(num_frames * test_fraction)
    valid_split_size = num_frames - train_split_size - test_split_size
    logger.debug(
        "Train split has {} items, test split has {} items, and validation "
        "split has {} items.",
        train_split_size,
        test_split_size,
        valid_split_size,
    )

    # Choose the frames to go in each split.
    np.random.shuffle(frame_nums)
    train_frames = frame_nums[0:train_split_size]
    test_frames = frame_nums[
        train_split_size : train_split_size + test_split_size
    ]
    valid_frames = frame_nums[train_split_size + test_split_size :]

    # Split the DataFrames.
    frame_num_column = local_annotations["frame_num"]
    train_data = local_annotations[frame_num_column.isin(train_frames)]
    test_data = local_annotations[frame_num_column.isin(test_frames)]
    valid_data = local_annotations[frame_num_column.isin(valid_frames)]

    return train_data, test_data, valid_data


def shuffle(annotations: pd.DataFrame) -> pd.DataFrame:
    """
    Randomly shuffles annotation data.

    Args:
        annotations: The annotations to shuffle, as a `DataFrame`.

    Returns:
        The shuffled annotations.

    """
    return annotations.sample(frac=1)


def _bytes_feature(
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


def _float_feature(feature_floats: Iterable[float]):
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


def _int_feature(feature_ints: Iterable[int]):
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


def _make_example(
    *,
    image: np.ndarray,
    frame_num: int,
    annotation_x: np.ndarray,
    annotation_y: np.ndarray,
) -> tf.train.Example:
    """
    Creates a single Tensorflow example.

    Args:
        image: The image data to include in the example.
        frame_num: The frame number associated with the image.
        annotation_x: The x-coordinates of the annotation points in the image.
        annotation_y: The y-coordinates of the annotation points in the image.

    Returns:
        The example that it created.

    """
    # Expand the frame number so it has the same shape as the others.
    frame_numbers = np.empty_like(annotation_x, dtype=np.int32)
    frame_numbers[:] = frame_num

    features = dict(
        image=_bytes_feature(image),
        annotation_x=_float_feature(annotation_x),
        annotation_y=_float_feature(annotation_y),
        frame_numbers=_int_feature(frame_numbers),
    )
    return tf.train.Example(features=tf.train.Features(feature=features))


def generate_tf_records(
    annotations: pd.DataFrame, cotton_images: CvatHandle
) -> Iterable[tf.train.Example]:
    """
    Generates TFRecord examples from the data.

    Args:
        annotations: The cleaned annotation data, as a Pandas `DataFrame`.
        cotton_images: The CVAT handle to use for retrieving image data.

    Yields:
        Each example that it produced.

    """
    frame_nums = annotations["frame_num"].unique()
    for frame_num in frame_nums:
        logger.debug("Generating example for frame {}.", frame_num)

        # Get all annotations for that frame.
        frame_annotations = annotations[annotations["frame_num"] == frame_num]
        # Extract the point columns.
        annotation_x = frame_annotations["x"].to_numpy()
        annotation_y = frame_annotations["y"].to_numpy()

        # Get the corresponding image.
        frame_image = cotton_images.get_frame(frame_num, compressed=True)

        # Normalize points to the shape of the image.
        frame_width, frame_height = cotton_images.get_frame_size(frame_num)
        annotation_x /= frame_width - 1
        annotation_y /= frame_height - 1

        yield _make_example(
            image=frame_image,
            frame_num=frame_num,
            annotation_x=annotation_x,
            annotation_y=annotation_y,
        )
