"""
Very simple utility for merging TFRecords files.
"""


from argparse import ArgumentParser
from pathlib import Path
from typing import List

import tensorflow as tf
from loguru import logger


def _merge_files(input_files: List[Path], *, output_file: Path) -> None:
    """
    Merges multiple TFRecords files with the same schema into one.

    Args:
        input_files: The input TFRecords files.
        output_file: The output file to write.

    """
    logger.info("Merging TFRecords files {}", input_files)

    input_files_str = [p.as_posix() for p in input_files]
    dataset = tf.data.TFRecordDataset(input_files_str)

    # Save the data.
    writer = tf.data.experimental.TFRecordWriter(output_file.as_posix())
    writer.write(dataset)


def _make_parser() -> ArgumentParser:
    """
    Returns:
        A parser to use for parsing the CLI arguments.

    """
    parser = ArgumentParser(description="Merge TFRecords files.")

    parser.add_argument("-i", "--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, default="merged.tfrecord")

    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    _merge_files(args.inputs, output_file=args.output)


if __name__ == "__main__":
    main()
