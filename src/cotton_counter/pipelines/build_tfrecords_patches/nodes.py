"""
Contains node definitions for the `build_tfrecords_patches` pipeline.
"""


import enum
from typing import Any, Iterable

import numpy as np
import tensorflow as tf
from loguru import logger
from pycvat import Task

from ..tfrecords_utils import bytes_feature, int_feature


@enum.unique
class AnnotationFilter(enum.IntEnum):
    """
    Specifies types of filtering that we can perform for annotations.
    """

    KEEP_POSITIVE = enum.auto()
    """
    Keep only positive examples.
    """
    KEEP_NEGATIVE = enum.auto()
    """
    Keep only negative examples.
    """
    KEEP_ALL = enum.auto()
    """
    Keep all examples.
    """


def _make_example(
    *, image: np.ndarray, frame_num: int, has_flower: bool
) -> tf.train.Example:
    """
    Creates a single Tensorflow example.

    Args:
        image: The image data to include in the example.
        frame_num: The frame number associated with the image.
        has_flower: Whether this image contains at least one flower.

    Returns:
        The example that it created.

    """
    features = dict(
        image=bytes_feature(image),
        frame_number=int_feature([frame_num]),
        # Boolean value is represented as an integer for Tensorflow.
        has_flower=int_feature([int(has_flower)]),
    )
    return tf.train.Example(features=tf.train.Features(feature=features))


def _generate_examples_from_task(
    task: Task,
    *,
    frame_offset: int,
    label_name: str,
    job_num: int = 0,
    keep_examples: AnnotationFilter = AnnotationFilter.KEEP_ALL
) -> Iterable[tf.train.Example]:
    """
    Generates TFRecord examples from a CVAT task.

    Args:
        task: The task to generate examples from.
        frame_offset: The initial frame number to use for these annotations.
        label_name: The name of the label that indicates whether we have a
            flower or not.
        job_num: The job number within that task to source annotations from.
        keep_examples: Specifies what filtering to perform on the
            annotations, if any.

    Yields:
        Each example that it produced.

    """
    # Get the job we are using.
    job = task.get_jobs()[job_num]

    # Get the label to use.
    flower_label = task.find_label(label_name)

    for frame_num in range(job.start_frame, job.end_frame + 1):
        # Get the annotations and filter to the relevant ones.
        annotations = job.annotations_for_frame(frame_num)
        flower_tags = list(
            filter(lambda a: a.label_id == flower_label.id, annotations)
        )
        if len(flower_tags) > 1:
            # This shouldn't happen, but it's not really fatal.
            logger.warning(
                "Frame {} from task {} has multiple flower tags.",
                frame_num,
                task.id,
            )
        has_flower = len(flower_tags) > 0

        if (
            has_flower and keep_examples == AnnotationFilter.KEEP_NEGATIVE
        ) or (
            not has_flower and keep_examples == AnnotationFilter.KEEP_POSITIVE
        ):
            # We should filter out this annotation.
            continue

        image = task.get_image(frame_num, compressed=True)

        yield _make_example(
            image=image,
            frame_num=frame_offset + frame_num,
            has_flower=len(flower_tags) > 0,
        )


def generate_tf_records(
    *,
    flower_label_name: str,
    keep_examples: AnnotationFilter = AnnotationFilter.KEEP_ALL,
    **tasks: Any
) -> Iterable[tf.train.Example]:
    """
    Generates TFRecord examples from multiple tasks on CVAT containing
    annotated patch data.

    Args:
        flower_label_name: The name of the label that indicates that a patch
            contains at least one flower.
        keep_examples: Specifies what filtering to perform on the
            annotations, if any.
        **tasks: The various CVAT tasks to aggregate data from. The names used
            for these keyword arguments do not matter.

    Yields:
        Each example that it produced.

    """
    frame_counter = 0

    for name, task in tasks.items():
        logger.info("Generating examples from {} (task {})...", name, task.id)
        task_examples = _generate_examples_from_task(
            task,
            frame_offset=frame_counter,
            label_name=flower_label_name,
            keep_examples=keep_examples,
        )

        for example in task_examples:
            yield example
            frame_counter += 1
