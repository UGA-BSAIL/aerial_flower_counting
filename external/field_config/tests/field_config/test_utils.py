"""
Tests for the `utils` module.
"""

from typing import Dict, Any

import geopandas as gpd

import pytest

from src.field_config import FieldConfig, label_plots
from src.field_config.field_config import RowDirection
from .data import EXAMPLE_PLOTS_1


@pytest.fixture
def example_1_plot_shapes() -> gpd.GeoDataFrame:
    """
    Loads the Example 1 plot boundaries.
    """
    return gpd.read_file(EXAMPLE_PLOTS_1)


@pytest.mark.parametrize(
    "row_direction",
    [RowDirection.NORTH_TO_SOUTH, RowDirection.WEST_TO_EAST],
    ids=["N-S", "W-E"],
)
def test_label_plots(
    example_field_1_yaml: Dict[str, Any],
    example_1_plot_shapes: gpd.GeoDataFrame,
    row_direction: RowDirection,
) -> None:
    """
    Tests that `label_plots` works.

    Args:
        example_field_1_yaml: The field config to test with.
        example_1_plot_shapes: The shapes of the plots in the field.
        row_direction: The direction of the rows in the plots.

    """
    # Arrange.
    # Set the row direction.
    if row_direction == RowDirection.NORTH_TO_SOUTH:
        example_field_1_yaml["row_direction"] = "north_to_south"
    elif row_direction == RowDirection.WEST_TO_EAST:
        example_field_1_yaml["row_direction"] = "west_to_east"

    field_config = FieldConfig.from_yml(example_field_1_yaml)

    # Act.
    labeled = list(
        label_plots(
            plot_boundaries=example_1_plot_shapes.geometry,
            field_config=field_config,
        )
    )

    # Assert.
    assert len(labeled) == len(example_1_plot_shapes)

    # Index by plot number.
    labeled_centroids = {p: b.centroid for b, p in labeled}

    if row_direction == RowDirection.NORTH_TO_SOUTH:
        # 1394 should be below 1393.
        assert labeled_centroids[1394].y < labeled_centroids[1393].y
        assert (labeled_centroids[1393].y - labeled_centroids[1394].y) > (
            labeled_centroids[1394].x - labeled_centroids[1393].x
        )
    if row_direction == RowDirection.WEST_TO_EAST:
        # 1394 should be right of 1393.
        assert labeled_centroids[1394].x > labeled_centroids[1393].x
        assert (labeled_centroids[1394].x - labeled_centroids[1393].x) > (
            labeled_centroids[1393].y - labeled_centroids[1394].y
        )
