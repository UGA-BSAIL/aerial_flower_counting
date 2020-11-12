"""
Very simple utility for merging TFRecords files.
"""


import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import tensorflow as tf
from loguru import logger


def _estimate_ratios(input_files: List[Path]) -> List[float]:
    """
    Estimates the relative amount of data in each of the input files.
    TensorFlow does not provide an easy way to query the number of elements
    in a TFRecord dataset, so this function works by simply looking at file
    sizes.

    Args:
        input_files: The input files we are merging.

    Returns:
        The fraction of the total data that we estimate is contained in each
        file.

    """
    # Get file sizes.
    file_sizes = [os.path.getsize(p) for p in input_files]
    logger.debug("Input file sizes: {}", file_sizes)

    # Normalize to fractions.
    total_size = sum(file_sizes)
    return [s / total_size for s in file_sizes]


def _merge_files(input_files: List[Path], *, output_file: Path) -> None:
    """
    Merges multiple TFRecords files with the same schema into one.

    Args:
        input_files: The input TFRecords files.
        output_file: The output file to write.

    """
    logger.info("Merging TFRecords files {}", input_files)

    # We try to distribute the data from each file uniformly.
    data_fractions = _estimate_ratios(input_files)
    logger.debug("Using fractions for sampling: {}", data_fractions)

    input_files_str = [p.as_posix() for p in input_files]
    input_datasets = [tf.data.TFRecordDataset([i]) for i in input_files_str]
    combined = tf.data.experimental.sample_from_datasets(
        input_datasets, data_fractions
    )

    # Save the data.
    writer = tf.data.experimental.TFRecordWriter(output_file.as_posix())
    writer.write(combined)


def _split_files(input_file: Path, *, output_files: List[Path]) -> None:
    """
    Splits a single TFRecords file evenly into multiple sub-files.

    Args:
        input_file: The input file to split.
        output_files: The output files to write.

    """
    num_outputs = len(output_files)
    logger.info(
        "Splitting TFRecords file {} into {} parts.", input_file, num_outputs,
    )

    # Load the input data.
    dataset = tf.data.TFRecordDataset([input_file.as_posix()])
    # Split the dataset.
    splits = [dataset.shard(num_outputs, i) for i in range(num_outputs)]

    # Create the writers for the output.
    writers = [
        tf.data.experimental.TFRecordWriter(o.as_posix()) for o in output_files
    ]
    # Write the data.
    for writer, split in zip(writers, splits):
        writer.write(split)


def _shuffle_file(
    input_file: Path, *, num_shuffles: int, num_splits: int = 10
) -> None:
    """
    Shuffles a file in-place through repeated splits and merges.

    Args:
        input_file: The input file to shuffle.
        num_shuffles: The number of times to split and merge.
        num_splits: Number of sub-files to split into on each iteration.

    """
    with TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        for i in range(num_shuffles):
            logger.info("Performing shuffle iteration {}.", i + 1)

            # Perform the split.
            split_files = [
                temp_dir / f"split_{s}.tfrecord" for s in range(num_splits)
            ]
            _split_files(input_file, output_files=split_files)

            # Perform the merge.
            input_file.unlink()
            _merge_files(split_files, output_file=input_file)


def _merge_command(args: Namespace) -> None:
    """
    Handler for the merge command.

    Args:
        args: The CLI arguments.

    """
    _merge_files(args.inputs, output_file=args.output)


def _split_command(args: Namespace) -> None:
    """
    Handler for the split command.

    Args:
        args: The CLI arguments.

    """
    _split_files(args.input, output_files=args.outputs)


def _shuffle_command(args: Namespace) -> None:
    """
    Handler for the shuffle command.

    Args:
        args: The CLI arguments.

    """
    _shuffle_file(
        args.input, num_shuffles=args.num_shuffles, num_splits=args.num_splits
    )


def _make_parser() -> ArgumentParser:
    """
    Returns:
        A parser to use for parsing the CLI arguments.

    """
    parser = ArgumentParser(
        description="Utilities for managing TFRecord files."
    )
    subparsers = parser.add_subparsers(
        help="Operation to perform.", required=True
    )

    merge_parser = subparsers.add_parser("merge", help="Merge multiple files.")
    merge_parser.add_argument(
        "-i", "--inputs", nargs="+", type=Path, required=True
    )
    merge_parser.add_argument(
        "-o", "--output", type=Path, default="merged.tfrecord"
    )
    merge_parser.set_defaults(func=_merge_command)

    split_parser = subparsers.add_parser("split", help="Split a single file.")
    split_parser.add_argument("-i", "--input", type=Path, required=True)
    split_parser.add_argument(
        "-o", "--outputs", nargs="+", type=Path, required=True
    )
    split_parser.set_defaults(func=_split_command)

    shuffle_parser = subparsers.add_parser(
        "shuffle", help="Shuffle a file " "in-place."
    )
    shuffle_parser.add_argument("-i", "--input", type=Path, required=True)
    shuffle_parser.add_argument(
        "-n",
        "--num-shuffles",
        type=int,
        default=5,
        help="Number of times to shuffle and merge.",
    )
    shuffle_parser.add_argument(
        "--num-splits",
        type=int,
        default=10,
        help="Number of splits to make each time.",
    )
    shuffle_parser.set_defaults(func=_shuffle_command)

    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
