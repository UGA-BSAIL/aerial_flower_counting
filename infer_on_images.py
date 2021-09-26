"""
Performs inference on specified local image_shape.
"""


from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tensorflow as tf
from loguru import logger

from src.cotton_counter.model.inference import (
    calculate_max_density,
    count_with_patches,
)
from src.cotton_counter.model.layers import CUSTOM_OBJECTS
from src.cotton_counter.model.patches import extract_standard_patches
from src.cotton_counter.model.visualization import visualize_heat_maps


def _make_density_map(
    *,
    model: tf.keras.Model,
    image: np.ndarray,
    patch_scale: float,
    patch_stride: float
) -> np.ndarray:
    """
    Performs inference on a given image, and creates the density map.

    Args:
        model: The model to use for inference.
        image: The image to create a density map for.
        patch_scale: The scale factor to apply to the patches we extract.
        patch_stride: The stride to use for extracting patches, provided in
            frame fractions like the scale.

    Returns:
        The density map, as a 4D tensor.

    """
    image_4d = tf.constant(np.expand_dims(image, 0))
    density_maps = count_with_patches(
        model, image_4d, patch_scale=patch_scale, patch_stride=patch_stride
    )

    return density_maps


def _infer_image(image: np.ndarray, patch_scale: float, **kwargs: Any) -> None:
    """
    Performs inference on an image and displays the results.

    Args:
        image: The image to create a density map for.
        patch_scale: The scale factor to apply to the patches we extract.
        **kwargs: Will be forwarded to `_make_density_map`.

    """
    density_map = _make_density_map(
        image=image, patch_scale=patch_scale, **kwargs
    )

    # Create a heatmap.
    max_density = calculate_max_density(image.shape, patch_scale=patch_scale)
    image_4d = tf.constant(np.expand_dims(image, 0))
    heatmaps = visualize_heat_maps(
        images=image_4d,
        features=tf.constant(density_map, dtype=tf.float32),
        max_color_threshold=max_density,
    )

    # Get the total count
    count_estimate = np.sum(density_map)

    # Display the results.
    logger.info("Total count: {}", count_estimate)
    # Reduce the size of the heatmaps for easier display.
    heatmap = heatmaps[0].numpy()
    heatmap = cv2.resize(heatmap, (1920, 1080))
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    cv2.imshow("Heatmap", heatmap)
    cv2.waitKey()


def _load_and_infer(cli_args: Namespace) -> None:
    """
    Loads an image and a model, performs inference on the image with the
    model, and displays the results.

    Args:
        cli_args: The parsed CLI arguments passed by the user.

    """
    # Load the model.
    logger.info("Loading model from {}...", cli_args.model)
    model = tf.keras.models.load_model(
        cli_args.model,
        custom_objects=dict(
            **CUSTOM_OBJECTS,
            extract_standard_patches=extract_standard_patches,
            tf=tf
        ),
        compile=False,
    )
    # Load the image.
    image = cv2.imread(cli_args.image.as_posix())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform inference.
    logger.info("Running inference...")
    _infer_image(
        model=model,
        image=image,
        patch_scale=cli_args.patch_scale,
        patch_stride=cli_args.patch_stride,
    )


def _make_parser() -> ArgumentParser:
    """
    Returns:
        A parser for parsing CLI arguments.

    """
    parser = ArgumentParser(
        description="Utility for performing inference on images."
    )

    parser.add_argument(
        "image",
        type=Path,
        help="The path to the image to perform inference on.",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        type=Path,
        help="The path to the model to use for inference.",
    )
    parser.add_argument(
        "-s",
        "--patch-scale",
        type=float,
        default=0.09,
        help="The patch scale to use during inference.",
    )
    parser.add_argument(
        "-t",
        "--patch-stride",
        type=float,
        default=0.03,
        help="The patch stride to use during inference.",
    )

    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    _load_and_infer(args)


if __name__ == "__main__":
    main()
