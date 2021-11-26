"""
Contains node definitions for the `build_tfrecords_patches` pipeline.
"""

import enum
from typing import Any, Dict, Iterable

import numpy as np
import tensorflow as tf
from loguru import logger
from pycvat import Job, Label, Task

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
    *, image: np.ndarray, frame_num: int, **kwargs: Any,
) -> tf.train.Example:
    """
    Creates a single Tensorflow example.

    Args:
        image: The image data to include in the example.
        frame_num: The frame number associated with the image.
        **kwargs: Will be parsed as additional boolean features for each
            example. The could be used to indicate the presence of an organ.

    Returns:
        The example that it created.

    """
    # Boolean values are represented as integers for Tensorflow.
    boolean_features = {k: int_feature([int(v)]) for k, v in kwargs.items()}

    features = dict(
        image=bytes_feature(image),
        frame_number=int_feature([frame_num]),
        **boolean_features,
    )
    return tf.train.Example(features=tf.train.Features(feature=features))


def _get_tags_for_frame(
    *, frame_num: int, job: Job, ids_to_labels: Dict[int, Label]
) -> Dict[str, bool]:
    """
    Extracts the boolean tag annotations for a particular frame.

    Args:
        frame_num: The frame number within the job.
        job: The job.
        ids_to_labels: Maps label IDs to allowed labels.

    Returns:
        A mapping of tag names to whether that tag is present or not.

    """
    # Get the annotations and filter to the relevant ones.
    annotations = job.annotations_for_frame(frame_num)

    annotation_label_ids = set()
    for annotation in annotations:
        label = ids_to_labels.get(annotation.label_id)
        if label is None:
            # Unknown label.
            logger.warning(
                "Ignoring unknown label ID {}. If this is "
                "intentional, it can be safely ignored.",
                annotation.label_id,
            )
            continue
        annotation_label_ids.add(label.id)

    # Convert labels to booleans.
    boolean_annotations = {}
    for label_id, label in ids_to_labels.items():
        if label_id in annotation_label_ids:
            boolean_annotations[label.name] = True
        else:
            boolean_annotations[label.name] = False
    logger.debug(
        "Boolean annotations for frame {}: {}", frame_num, boolean_annotations,
    )

    return boolean_annotations


def _generate_examples_from_task(
    task: Task,
    *,
    frame_offset: int,
    label_names: Iterable[str],
    job_num: int = 0,
    keep_examples: AnnotationFilter = AnnotationFilter.KEEP_ALL,
) -> Iterable[tf.train.Example]:
    """
    Generates TFRecord examples from a CVAT task.

    Args:
        task: The task to generate examples from.
        frame_offset: The initial frame number to use for these annotations.
        label_names: The names of the labels that we want to include in our
            examples.
        job_num: The job number within that task to source annotations from.
        keep_examples: Specifies what filtering to perform on the
            annotations, if any.

    Yields:
        Each example that it produced.

    """
    # Get the job we are using.
    job = task.jobs[job_num]

    # Get the label to use.
    possible_labels = [task.find_label(label) for label in label_names]
    # Map unique label IDs to labels.
    ids_to_labels = {label.id: label for label in possible_labels}

    for frame_num in range(job.start_frame, job.end_frame + 1):
        boolean_annotations = _get_tags_for_frame(
            frame_num=frame_num, job=job, ids_to_labels=ids_to_labels
        )

        # Possibly filter out positive or negative examples.
        has_any_label = True in boolean_annotations.values()
        if (
            has_any_label and keep_examples == AnnotationFilter.KEEP_NEGATIVE
        ) or (
            not has_any_label
            and keep_examples == AnnotationFilter.KEEP_POSITIVE
        ):
            # We should filter out this annotation.
            continue

        image = task.get_image(frame_num, compressed=True)

        yield _make_example(
            image=image,
            frame_num=frame_offset + frame_num,
            **boolean_annotations,
        )


def generate_tf_records(
    *,
    flower_label_name: str,
    keep_examples: AnnotationFilter = AnnotationFilter.KEEP_ALL,
    **tasks: Any,
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
            label_names=[flower_label_name],
            keep_examples=keep_examples,
        )

        for example in task_examples:
            yield example
            frame_counter += 1
