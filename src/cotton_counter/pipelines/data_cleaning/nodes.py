"""
Contains node definitions for the `data_cleaning` pipeline.
"""


import itertools
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from pycvat import LabeledShape, Task

from ...cvat_utils import get_main_job


def remove_unannotated(cvat_task: Task) -> List[Tuple[int, List]]:
    """
    Filters out data that has no annotations.

    Args:
        cvat_task: The raw CVAT task to filter data from.

    Returns:
        List of tuples of the frame number in the raw CVAT data and the
        corresponding annotations for that frame.

    """
    job = get_main_job(cvat_task)
    logger.debug(
        "Have {} raw examples in dataset.", job.end_frame - job.start_frame
    )

    filtered_annotations = []
    for count, annotations in enumerate(job.iter_annotations()):
        if len(annotations) != 0:
            # We have annotations.
            filtered_annotations.append((count, annotations))

    return filtered_annotations


def _remove_with_attribute(
    *,
    annotations: Iterable[Tuple[int, List]],
    cvat_task: Task,
    attribute_name: str,
    attribute_value: Any,
) -> Iterable[Tuple[int, List]]:
    """
    Removes any annotations that have a specific attribute.

    Args:
        annotations: The annotations to convert. It expects the output from the
            `remove_unannotated` node.
        cvat_task: The CVAT task that these annotations came from.
        attribute_name: The name of the attribute to look at.
        attribute_value: The value of this attribute to remove annotations with.

    Yields:
        Same annotations, but excluding ones with this attribute set to this
        value.

    """
    # Find the corresponding annotation IDs.
    labels = cvat_task.get_labels()
    attributes = itertools.chain.from_iterable(
        (la.attributes for la in labels)
    )
    for attribute in attributes:
        if attribute.name == attribute_name:
            attribute_id = attribute.id
            break
    else:
        # In this case, we have no such attribute, so there is no filtering
        # to do.
        logger.debug(
            "Attribute '{}' is not present in any task labels.", attribute_name
        )
        yield from annotations
        return

    def has_filtered_attribute(_annotation: LabeledShape) -> bool:
        """
        Args:
            _annotation: The annotation to check.

        Returns:
            True if the annotation contains an attribute we are filtering.

        """
        for _attribute in _annotation.attributes:
            if (
                _attribute.spec_id == attribute_id
                and _attribute.value == attribute_value
            ):
                return True
        return False

    # Remove annotations with the correct attribute value.
    for frame_num, frame_annotations in annotations:
        filtered_annotations = []
        for annotation in frame_annotations:
            if not has_filtered_attribute(annotation):
                filtered_annotations.append(annotation)
        logger.debug(
            "Filtered out {} annotations from frame {}.",
            len(frame_annotations) - len(filtered_annotations),
            frame_num,
        )
        yield frame_num, filtered_annotations


def remove_with_attributes(
    *,
    annotations: Iterable[Tuple[int, List]],
    cvat_task: Task,
    attributes: Dict[str, Any],
) -> List[Tuple[int, List]]:
    """
    Removes any annotations that have specific attributes.

    Args:
        annotations: The annotations to convert. It expects the output from the
            `remove_unannotated` node.
        cvat_task: The CVAT task that these annotations came from.
        attributes: A dictionary mapping attribute names to attribute values. If
            an annotation has any of the named attributes, and they have the
            corresponding value, they will be removed.

    Returns:
        Same annotations, but excluding ones with any of these attributes set to
        these values.

    """
    filtered_annotations = annotations
    for name, value in attributes.items():
        logger.debug(
            "Removing annotations with attribute {} set to {}.", name, value
        )
        filtered_annotations = _remove_with_attribute(
            annotations=filtered_annotations,
            cvat_task=cvat_task,
            attribute_name=name,
            attribute_value=value,
        )

    return list(filtered_annotations)


def annotations_as_dataframe(
    annotations: List[Tuple[int, List]]
) -> pd.DataFrame:
    """
    Converts annotations to a Pandas `DataFrame`.

    Args:
        annotations: The annotations to convert. It expects the output from the
            `remove_unannotated` node.

    Returns:
        The annotation data, as a `DataFrame`.

    """
    per_frame_dfs = []

    for frame_num, point_annotations in annotations:
        # Flatten the points.
        points = [a.points for a in point_annotations]
        flat_points_for_frame = list(itertools.chain.from_iterable(points))
        flat_points_for_frame = np.array(flat_points_for_frame)

        # Split out the x and y coordinates of the points.
        x_values = flat_points_for_frame[::2]
        y_values = flat_points_for_frame[1::2]
        assert len(x_values) == len(
            y_values
        ), "Not all points have both x and y values."

        # We want the frame numbers to correspond exactly with the flattened
        # points.
        frame_numbers_for_frame = [frame_num] * len(x_values)

        # Create the DataFrame.
        data_frame = pd.DataFrame(
            data={
                "frame_num": frame_numbers_for_frame,
                "x": x_values,
                "y": y_values,
            }
        )
        per_frame_dfs.append(data_frame)

    # Merge all the DataFrames into one for all the frames.
    merged_data_frame = pd.concat(per_frame_dfs, ignore_index=True)
    # Add the frame number as a secondary index.
    merged_data_frame.set_index("frame_num", append=True)

    return merged_data_frame


def get_frame_sizes(
    annotations_frame: pd.DataFrame, cvat_task: Task
) -> pd.DataFrame:
    """
    Gets a `DataFrame` containing the sizes in pixels of each frame. Each row
    should correspond exactly to the same row in the annotations `DataFrame`.

    Args:
        annotations_frame: The `DataFrame` containing annotations.
        cvat_task: The raw CVAT task handle.

    Returns:
        A `DataFrame` containing columns for the width and height of each
        frame image.

    """
    # Get the frame sizes.
    frame_sizes = [
        cvat_task.get_image_size(n) for n in annotations_frame["frame_num"]
    ]
    # Coerce into its own data frame.
    return pd.DataFrame(
        data=np.array(frame_sizes), columns=["width", "height"]
    )


def drop_out_of_bounds(
    annotations_frame: pd.DataFrame, frame_sizes: pd.DataFrame
) -> pd.DataFrame:
    """
    Removes annotations that are outside the bounds of frame images.

    Args:
        annotations_frame: The `DataFrame` containing annotations.
        frame_sizes: The `DataFrame` containing the corresponding size of
            each frame image.

    Returns:
        A version of `annotations_frame` with out-of-bounds data removed.

    """
    out_of_bounds_x = (annotations_frame["x"] < 0) | (
        annotations_frame["x"] >= frame_sizes["width"]
    )
    out_of_bounds_y = (annotations_frame["y"] < 0) | (
        annotations_frame["y"] >= frame_sizes["height"]
    )

    return annotations_frame[~(out_of_bounds_x | out_of_bounds_y)]
