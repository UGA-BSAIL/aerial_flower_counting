"""
Represents the configuration of a field.
"""

from typing import Iterable, Dict, Any
from functools import singledispatch
from tabulate import tabulate

from .field_row import FieldRow


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

    def draw(self) -> None:
        """
        Prints a visual representation of the field.

        """
        table = []
        for row_i, row in enumerate(self.__rows):
            table_row = []
            for plot_i in range(len(row)):
                table_row.append(self.get_plot_num(row_i, plot_i))
            table.append(table_row)

        print(tabulate(table, tablefmt="simple_grid"))
