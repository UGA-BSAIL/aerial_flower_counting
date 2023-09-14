from pathlib import Path
from typing import List
import argparse


def _find_images(image_dir: Path) -> List[Path]:
    """
    Finds all images in the given directory.

    Args:
        image_dir: The directory to search in.

    Returns:
        A list of paths to the images.
    """
    image_suffixes = {".png", ".jpg", ".jpeg"}
    image_suffixes.update({suffix.upper() for suffix in image_suffixes})
    return [
        path for path in image_dir.iterdir() if path.suffix in image_suffixes
    ]


def _add_missing_labels(image_dir: Path) -> None:
    """
    Adds the missing label files to an image directory.
    
    Args:
        image_dir: The image directory.

    """
    for image_path in _find_images(image_dir):
        label_path = image_path.with_suffix(".txt")
        if not label_path.exists():
            label_path.write_text("")


def _parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        "Create manifest files for YOLO datasets."
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="The directory containing the images.",
    )
    parser.add_argument(
        "-b",
        "--base-dir",
        type=Path,
        default=Path("."),
        help="All saved paths will be relative to this directory.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        default=Path("train.txt"),
        help="The file to write the manifest to.",
    )
    return parser.parse_args()


def main() -> None:
    cli_args = _parse_args()

    image_paths = _find_images(cli_args.image_dir)
    _add_missing_labels(cli_args.image_dir)
    image_paths = [path.relative_to(cli_args.base_dir) for path in image_paths]
    cli_args.output_file.write_text(
        "\n".join([p.as_posix() for p in image_paths])
    )


if __name__ == "__main__":
    main()
