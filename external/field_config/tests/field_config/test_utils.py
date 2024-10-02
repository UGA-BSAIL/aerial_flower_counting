"""
Tests for the `utils` module.
"""

import geopandas as gpd

import pytest

from src.field_config import FieldConfig, label_plots, RowDirection
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
    example_field_1: FieldConfig,
    example_1_plot_shapes: gpd.GeoDataFrame,
    row_direction: RowDirection,
) -> None:
    """
    Tests that `label_plots` works.

    Args:
        example_field_1: The field config to test with.
        example_1_plot_shapes: The shapes of the plots in the field.
        row_direction: The direction of the rows in the plots.

    """
    # Act.
    labeled = list(
        label_plots(
            plot_boundaries=example_1_plot_shapes.geometry,
            field_config=example_field_1,
            row_direction=row_direction,
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