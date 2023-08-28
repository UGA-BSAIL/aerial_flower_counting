"""
Represents the configuration of a field.
"""


import numpy as np
from typing import Iterable, Dict, Any
from functools import singledispatch


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


class FieldConfig:
    """
    Represents the configuration of a field.
    """

    def __init__(self, rows: Iterable[FieldRow]):
        """
        Args:
            rows: The rows in this field.

        """
        self.__rows = list(rows)

    @classmethod
    def from_yml(cls, config: Dict[str, Any]) -> "FieldConfig":
        """
        Creates a field configuration based on a YAML specification.

        Args:
            config: The specification from the YAML file.

        Returns:
            The `FieldConfig` it created.

        """
        rows_by_name = {}

        @singledispatch
        def _parse_row(spec: Any, name: str) -> FieldRow:
            raise ValueError(f"Unknown row type: {spec}")

        @_parse_row.register
        def _(spec: int, name: str) -> FieldRow:
            return FieldRow.from_number(name, spec)

        @_parse_row.register
        def _(spec: dict, name: str) -> FieldRow:
            if "range" in spec.keys():
                # This is a range row.
                return FieldRow.from_range(
                    name,
                    start=spec["range"]["start"],
                    end=spec["range"]["end"],
                    repeats=spec["range"].get("repeats", 1),
                )
            elif "shift" in spec.keys():
                # This is a shifted row.
                source_row = rows_by_name[spec["shift"]["row"]]
                return source_row.clone_shifted(spec["shift"]["amount"])

        @_parse_row.register
        def _(spec: list, name: str) -> FieldRow:
            # Parse all the sub-rows and then merge them.
            return FieldRow.merge(*[_parse_row(row, name) for row in spec])

        for row_name, row_spec in config["rows"].items():
            rows_by_name[row_name] = _parse_row(row_spec, row_name)

        return cls(rows_by_name.values())

    def get_plot_num(self, row_index: int, plot_index: int) -> int:
        """
        Gets the plot number for the given row and plot index.

        Args:
            row_index: The row index.
            plot_index: The plot index.

        Returns:
            The plot number.

        """
        return self.__rows[row_index].get_plot_num(plot_index)

    def get_plot_num_row_major(self, index: int) -> int:
        """
        Gets the plot number for the given row-major index.

        Args:
            index: The row-major index.

        Returns:
            The plot number.

        """
        total_num_plots = 0
        for row_index in range(len(self.__rows)):
            if total_num_plots + len(self.__rows[row_index]) > index:
                # We found the correct row.
                plot_index = index - total_num_plots
                break
            total_num_plots += len(self.__rows[row_index])
        else:
            # Index is too large.
            raise IndexError(f"Index {index} is out of range.")

        return self.get_plot_num(row_index, plot_index)

    @property
    def num_rows(self) -> int:
        """
        Gets the number of rows in this field.

        Returns:
            The number of rows in this field.

        """
        return len(self.__rows)

    @property
    def num_plots(self) -> int:
        """
        Gets the number of plots in this field.

        Returns:
            The number of plots in this field.

        """
        return sum(len(row) for row in self.__rows)
