"""
Utility functions.
"""

import enum
from functools import reduce
from typing import Iterable, Tuple

from more_itertools import batched
from shapely.geometry import Polygon

from .field_config import FieldConfig


@enum.unique
class RowDirection(enum.IntEnum):
    """
    Represents possible directions for the rows.
    """

    NORTH_TO_SOUTH = enum.auto()
    WEST_TO_EAST = enum.auto()


def label_plots(
    *,
    plot_boundaries: Iterable[Polygon],
    field_config: FieldConfig,
    row_direction: RowDirection = RowDirection.NORTH_TO_SOUTH,
) -> Iterable[Tuple[Polygon, int]]:
    """
    Computes the plot number for each plot in a shapefile.

    Args:
        plot_boundaries: The boundaries of the plots.
        field_config: The configuration of the field.
        row_direction: The direction that the rows in the field are pointing.

    Yields:
        Each plot boundary polygon, along with the corresponding plot number,
        in the original order that the plot boundaries were provided in.

    """
    # Group the plots into rows.
    plot_boundaries = list(plot_boundaries)
    plot_centers = [p.centroid for p in plot_boundaries]
    boundaries_with_centers = [
        (b, c, i)
        for i, (b, c) in enumerate(zip(plot_boundaries, plot_centers))
    ]
    if row_direction == RowDirection.NORTH_TO_SOUTH:
        # Sort by x coordinate.
        boundaries_with_centers_sorted_plot = sorted(
            boundaries_with_centers, key=lambda p: p[1].x
        )
    else:
        # Sort by y coordinate.
        boundaries_with_centers_sorted_plot = sorted(
            boundaries_with_centers, key=lambda p: p[1].y, reverse=True
        )
    boundaries_with_centers_by_row = list(
        batched(
            boundaries_with_centers_sorted_plot,
            field_config.num_plots // field_config.num_rows,
        )
    )

    # Sort by plot number within rows.
    if row_direction == RowDirection.NORTH_TO_SOUTH:
        boundaries_with_centers_sorted = [
            sorted(row, key=lambda p: p[1].y, reverse=True)
            for row in boundaries_with_centers_by_row
        ]
    else:
        boundaries_with_centers_sorted = [
            sorted(row, key=lambda p: p[1].x)
            for row in boundaries_with_centers_by_row
        ]
    # Unbatch
    boundaries_with_centers_sorted = reduce(
        lambda x, y: x + y, boundaries_with_centers_sorted, []
    )
    sorted_boundary_indices = [i for _, _, i in boundaries_with_centers_sorted]

    # Assign real plot numbers to them.
    boundary_plot_nums = {
        i: field_config.get_plot_num_row_major(j)
        for j, i in enumerate(sorted_boundary_indices)
    }
    for i, boundary in enumerate(plot_boundaries):
        yield boundary, boundary_plot_nums[i]
