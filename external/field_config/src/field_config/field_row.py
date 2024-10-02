"""
Represents a single row in the field.
"""


import numpy as np


class FieldRow:
    """
    Represents a single row in the field.
    """

    def __init__(self, row_name: str, *, plot_numbers: np.array):
        """
        Args:
            row_name: The name of this row.
            plot_numbers: The plot numbers for each plot in this row.

        """
        self.__name = row_name
        self.__plot_numbers = plot_numbers

    @classmethod
    def from_range(
        cls, name: str, *, start: int, end: int, repeats: int = 1
    ) -> "FieldRow":
        """
        Creates a row from the given range of plot numbers.

        Args:
            name: The name of the row.
            start: The starting plot number.
            end: The ending plot number.
            repeats: The number of times each plot number repeats.

        Returns:
            The created row.

        """
        plot_numbers = np.arange(min(start, end), max(start, end) + 1)
        # It is valid to have start > end, which means we want to count
        # backwards.
        if start > end:
            plot_numbers = plot_numbers[::-1]
        # Add repeats.
        plot_numbers = np.repeat(plot_numbers, repeats)

        return FieldRow(name, plot_numbers=plot_numbers)

    @classmethod
    def from_number(cls, name: str, plot_num: int) -> "FieldRow":
        """
        Creates a row with a single plot number.

        Args:
            name: The name of the row.
            plot_num: The plot number.

        Returns:
            The created row.

        """
        return FieldRow(name, plot_numbers=np.array([plot_num]))

    def __len__(self) -> int:
        """
        Gets the number of plots in this row.

        Returns:
            The number of plots in this row.

        """
        return len(self.__plot_numbers)

    def get_plot_num(self, plot_index: int) -> int:
        """
        Gets the plot number for the given plot index.

        Args:
            plot_index: The plot index.

        Returns:
            The plot number.

        """
        return self.__plot_numbers[plot_index]

    def clone_shifted(self, shift: int) -> "FieldRow":
        """
        Clones this row with the given shift for the plot numbers.

        Args:
            shift: The shift to apply to the plot numbers.

        Returns:
            The cloned row.

        """
        return FieldRow(self.__name, plot_numbers=self.__plot_numbers + shift)

    @classmethod
    def merge(cls, *rows: "FieldRow") -> "FieldRow":
        """
        Merges the given rows into a single row.

        Args:
            *rows: The rows to merge, in the order that they will be merged.

        Returns:
            The merged row.

        """
        plot_numbers = np.concatenate([row.__plot_numbers for row in rows])
        return FieldRow(rows[0].__name, plot_numbers=plot_numbers)
