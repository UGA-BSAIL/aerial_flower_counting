from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


def _fix_one(label_file: Path) -> None:
    label_data = pd.read_csv(
        label_file, sep=" ", names=["class", "extra", "b1", "b2", "b3", "b4"]
    )

    # Remove the extra column.
    label_data.drop("extra", axis=1, inplace=True)
    # Make sure classes are all zero.
    label_data["class"] = 0

    # Write back to the file.
    label_data.to_csv(label_file, sep=" ", header=False, index=False)


def _fix_all(label_dir: Path) -> None:
    for label_file in label_dir.iterdir():
        _fix_one(label_file)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("label_dir", type=Path, help="Label directory")
    cli_args = parser.parse_args()

    _fix_all(cli_args.label_dir)


if __name__ == "__main__":
    main()
