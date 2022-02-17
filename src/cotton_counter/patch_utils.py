"""
Utility for extracting patches from images.
"""


from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from loguru import logger

from .model.patches import extract_standard_patches


def _iter_images(image_dir: Path) -> Iterable[Tuple[str, np.ndarray]]:
    """
    Iterates through the images in a directory, skipping any non-image files.

    Args:
        image_dir: The image directory.

    Yields:
        The name of the image file, along with the actual image data.

    """
    for item in image_dir.iterdir():
        if not item.is_file():
            logger.debug("Skipping item {} which is not a file.", item.name)
            continue

        # Try to load it as an image.
        image = cv2.imread(item.as_posix())
        if image is None:
            logger.debug("Skipping item {} which is not an image.", item.name)
            continue

        yield item.name, image


def _extract_all_patches(
    *,
    input_dir: Path,
    output_dir: Path,
    patch_scale: float,
    patch_stride: Optional[float] = None,
) -> None:
    """
    Extracts patches from all the images in `input_dir` and saves them
    to `output_dir`.

    Args:
        input_dir: The input directory containing images.
        output_dir: The output directory to save patches to.
        patch_scale: The scale of the patches to extract.
        patch_stride: The stride of the patches to extract. If not specified,
            it will be the same as the scale. (No overlap.)

    """
    if patch_stride is None:
        # Don't use any overlap.
        patch_stride = patch_scale

    # Create output directory if needed.
    output_dir.mkdir(exist_ok=True)

    for image_name, image in _iter_images(input_dir):
        # Extract the patches.
        image_4d = tf.expand_dims(image, axis=0)
        patches = extract_standard_patches(
            image_4d, patch_scale=patch_scale, patch_stride=patch_stride
        ).numpy()

        # Save the patches.
        for i, patch in enumerate(patches):
            patch_path = output_dir / f"p{i}_{image_name}"
            logger.info("Saving patch to {}.", patch_path)
            cv2.imwrite(patch_path.as_posix(), patch)


def _make_parser() -> ArgumentParser:
    """
    Creates the parser to use for parsing CLI arguments.

    Returns:
        The parser that it created.

    """
    parser = ArgumentParser(description="Extracts patches from images.")

    parser.add_argument(
        "image_dir", type=Path, help="Directory containing input images."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("patches"),
        help="Directory to save output patches to.",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=float,
        required=True,
        help="Patch scale to use, in frame fractions.",
    )
    parser.add_argument(
        "-t",
        "--stride",
        type=float,
        help="Patch stride to use, in frame fractions.",
    )

    return parser


def main() -> None:
    parser = _make_parser()
    cli_args = parser.parse_args()

    _extract_all_patches(
        input_dir=cli_args.image_dir,
        output_dir=cli_args.output_dir,
        patch_scale=cli_args.scale,
        patch_stride=cli_args.stride,
    )


if __name__ == "__main__":
    main()
