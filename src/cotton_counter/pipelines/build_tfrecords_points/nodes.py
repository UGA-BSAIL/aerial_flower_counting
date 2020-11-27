"""
Contains node definitions for the `build_tfrecords_points` pipeline.
"""


from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from pycvat import Task

from ..tfrecords_utils import bytes_feature, float_feature, int_feature


def make_splits(
    local_annotations: pd.DataFrame, split_fractions: Tuple[float, ...],
) -> Tuple[pd.DataFrame, ...]:
    """
    Generates the train, test and validation splits for the data.

    Args:
        local_annotations: The `DataFrame` containing cleaned annotation data.
        split_fractions: The fraction of the data that will go in each generated
            split. Should add up to 1.0.

    Returns:
        Annotations for each split corresponding to `split_fractions`,
        as a separate `DataFrame`.

    """
    # We split on frames instead of points so we don't have data from the
    # same frame spread across multiple splits.
    frame_nums = local_annotations["frame_num"].unique()
    num_frames = len(frame_nums)
    logger.debug("Splitting data from {} frames.", num_frames)

    split_fractions = np.array(split_fractions)
    split_sizes = split_fractions * num_frames
    split_sizes = split_sizes.astype(np.int32)
    logger.debug("Number of items in each split: {}", split_sizes)

    # Choose the frames to go in each split.
    np.random.shuffle(frame_nums)
    split_frame_indices = []
    split_start = 0
    for split_size in split_sizes:
        split_end = split_start + split_size
        frame_indices = frame_nums[split_start:split_end]
        split_frame_indices.append(frame_indices)

        split_start += split_size

    # Split the DataFrames.
    frame_num_column = local_annotations["frame_num"]
    all_split_data = []
    for frame_indices in split_frame_indices:
        split_data = local_annotations[frame_num_column.isin(frame_indices)]
        all_split_data.append(split_data)

    return tuple(all_split_data)


def shuffle(annotations: pd.DataFrame) -> pd.DataFrame:
    """
    Randomly shuffles annotation data.

    Args:
        annotations: The annotations to shuffle, as a `DataFrame`.

    Returns:
        The shuffled annotations.

    """
    return annotations.sample(frac=1)


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
        image=bytes_feature(image),
        annotation_x=float_feature(annotation_x),
        annotation_y=float_feature(annotation_y),
        frame_numbers=int_feature(frame_numbers),
    )
    return tf.train.Example(features=tf.train.Features(feature=features))


def generate_tf_records(
    annotations: pd.DataFrame, cotton_images: Task
) -> Iterable[tf.train.Example]:
    """
    Generates TFRecord examples from the data.

    Args:
        annotations: The cleaned annotation data, as a Pandas `DataFrame`.
        cotton_images: The CVAT task handle to use for retrieving image data.

    Yields:
        Each example that it produced.

    """
    frame_nums = annotations["frame_num"].unique().tolist()
    for frame_num in frame_nums:
        logger.debug("Generating example for frame {}.", frame_num)

        # Get all annotations for that frame.
        frame_annotations = annotations[annotations["frame_num"] == frame_num]
        # Extract the point columns.
        annotation_x = frame_annotations["x"].to_numpy()
        annotation_y = frame_annotations["y"].to_numpy()

        # Get the corresponding image.
        frame_image = cotton_images.get_image(frame_num, compressed=True)

        # Normalize points to the shape of the image.
        frame_width, frame_height = cotton_images.get_image_size(frame_num)
        annotation_x /= frame_width - 1
        annotation_y /= frame_height - 1

        yield _make_example(
            image=frame_image,
            frame_num=frame_num,
            annotation_x=annotation_x,
            annotation_y=annotation_y,
        )
