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
    predict_with_activation_maps_patched,
)
from src.cotton_counter.model.layers import CUSTOM_OBJECTS
from src.cotton_counter.model.patches import extract_standard_patches
from src.cotton_counter.model.visualization import visualize_heat_maps


def _compute_activations(
    *, model: tf.keras.Model, image: np.ndarray
) -> tf.Tensor:
    """
    Computes the raw activations from running the model on an image.

    Args:
        model: The model to run.
        image: The image to run it on.

    Returns:
        The activations.

    """
    image_4d = tf.constant(np.expand_dims(image, 0))
    return predict_with_activation_maps_patched(model, image_4d)


def _visualize_activations(
    *, image: np.ndarray, activation_maps: tf.Tensor
) -> tf.Tensor:
    """
    Visualizes activation maps from the model.

    Args:
        image: The image that the activation corresponds to.
        activation_maps: The corresponding activation, as a 4D tensor.

    Returns:
        The visualized activations, as images.

    """
    image_4d = tf.constant(np.expand_dims(image, 0))

    # We want everything to be positive in order to visualize correctly.
    min_activation = tf.reduce_min(activation_maps)
    activation_maps -= tf.where(min_activation < 0, min_activation, 0)
    max_activation = tf.reduce_max(activation_maps)

    # Since the zero class is actually positive, to make the visualization
    # more intuitive, we reverse the activations.
    activation_maps = max_activation - activation_maps

    return visualize_heat_maps(
        images=image_4d,
        features=activation_maps,
        max_color_threshold=max_activation,
    )


def _make_density_map(
    *, activation_maps: tf.Tensor, patch_scale: float, patch_stride: float
) -> np.ndarray:
    """
    Performs inference on a given image, and creates the density map.

    Args:
        activation_maps: The raw activation maps for an image.
        patch_scale: The scale factor to apply to the patches we extract.
        patch_stride: The stride to use for extracting patches, provided in
            frame fractions like the scale.

    Returns:
        The density map, as a 4D tensor.

    """
    density_maps = count_with_patches(
        activation_maps, patch_scale=patch_scale, patch_stride=patch_stride
    )

    return density_maps


def _display_image(image_tensor: tf.Tensor, *, window_name: str) -> None:
    """
    Displays an image on the screen.

    Args:
        image_tensor: The image to display, as a 4D tensor.
        window_name: The name of the display window to use.

    """
    image = image_tensor[0].numpy()
    image = cv2.resize(image, (1920, 1080))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, image)


def _infer_image(
    model: tf.keras.Model, image: np.ndarray, patch_scale: float, **kwargs: Any
) -> None:
    """
    Performs inference on an image and displays the results.

    Args:
        model: The model to use for inference.
        image: The image to create a density map for.
        patch_scale: The scale factor to apply to the patches we extract.
        **kwargs: Will be forwarded to `_make_density_map`.

    """
    activation_maps = _compute_activations(model=model, image=image)
    density_map = _make_density_map(
        activation_maps=activation_maps, patch_scale=patch_scale, **kwargs
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

    # Visualize the raw activations.
    activation_viz = _visualize_activations(
        image=image, activation_maps=activation_maps
    )

    # Display the results.
    logger.info("Total count: {}", count_estimate)
    _display_image(heatmaps, window_name="Heatmap")
    _display_image(activation_viz, window_name="Activation")
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
