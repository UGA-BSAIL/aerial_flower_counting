"""
Contains node definitions for the `data_cleaning` pipeline.
"""


from typing import List, Tuple

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
