"""
Tests for the `field_config` module.
"""

from typing import Dict, Any

import pytest

from src.field_config import FieldConfig, RowDirection


def test_field_size(example_field_1: FieldConfig) -> None:
    """
    Tests that it gets the correct size for the field.

    Args:
        example_field_1: The field configuration.

    """
    # Assert.
    assert example_field_1.num_rows == 6
    assert example_field_1.num_plots == 100 * 6


def test_get_plot_num(example_field_1: FieldConfig) -> None:
    """
    Tests that it gets the correct plot number for a given row and column.

    Args:
        example_field_1: The field configuration.

    """
    # Assert.
    assert example_field_1.get_plot_num(2, 6) == 1526
    assert example_field_1.get_plot_num(4, 41) == 1673
    assert example_field_1.get_plot_num(5, 72) == 1742


def test_get_plot_num_row_major(example_field_1: FieldConfig) -> None:
    """
    Tests that it gets the correct plot number using row-major indexing.

    Args:
        example_field_1: The field configuration.

    """
    # Assert.
    assert example_field_1.get_plot_num_row_major(206) == 1526
    assert example_field_1.get_plot_num_row_major(441) == 1673
    assert example_field_1.get_plot_num_row_major(572) == 1742


@pytest.mark.parametrize(
    "direction",
    [RowDirection.NORTH_TO_SOUTH, RowDirection.WEST_TO_EAST],
    ids=["N-S", "W-E"],
)
def test_row_direction(
    example_field_1_yaml: Dict[str, Any], direction: RowDirection
) -> None:
    """
    Tests that the row direction is correct.
    
    Args:
        example_field_1_yaml: The field configuration.
        direction: The row direction to use for testing.

    """
    # Arrange.
    # Set the row direction.
    if direction == RowDirection.NORTH_TO_SOUTH:
        example_field_1_yaml["row_direction"] = "north_to_south"
    elif direction == RowDirection.WEST_TO_EAST:
        example_field_1_yaml["row_direction"] = "west_to_east"

    # Act.
    field_config = FieldConfig.from_yml(example_field_1_yaml)

    # Assert.
    assert field_config.row_direction == direction
