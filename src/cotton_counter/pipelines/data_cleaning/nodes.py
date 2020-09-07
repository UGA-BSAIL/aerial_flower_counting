"""
Contains node definitions for the `data_cleaning` pipeline.
"""


import itertools
from typing import List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from pycvat.dataset import CvatHandle


def remove_unannotated(cvat_data: CvatHandle) -> List[Tuple[int, List]]:
    """
    Filters out data that has no annotations.

    Args:
        cvat_data: The raw CVAT dataset to filter.

    Returns:
        List of tuples of the frame number in the raw CVAT data and the
        corresponding annotations for that frame.

    """
    logger.debug("Have {} raw examples in dataset.", cvat_data.num_frames)

    filtered_annotations = []
    for count, annotations in enumerate(cvat_data.iter_annotations()):
        if len(annotations) != 0:
            # We have annotations.
            filtered_annotations.append((count, annotations))

    return filtered_annotations


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
