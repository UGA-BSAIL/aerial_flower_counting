import tensorflow as tf


def _colorize_heat_maps(
    features: tf.Tensor, *, max_color_threshold: float
) -> tf.Tensor:
    """
    Converts 2D float matrices to a false-color heat map to make them easier to
    view by humans.

    Args:
        features: The 4D tensor of features to visualize.
        max_color_threshold: Threshold to use for colorization.
            Any pixel with this value or more will show up as the hottest color.

    Returns:
        The colorized heat maps. Note that these will be returned as uint8
        images, ready for display.

    """
    max_color_threshold = tf.constant(max_color_threshold, dtype=tf.float32)

    # Low values should show up as blue.
    blue_channel = max_color_threshold - features
    # Medium values should show up as yellow.
    max_green_point = max_color_threshold / tf.constant(2.0)
    green_channel = (
        tf.constant(-2.0) * tf.abs(features - max_green_point)
        + max_color_threshold
    )
    # High values should show up as red.
    red_channel = tf.constant(2.0) * (features - max_green_point)

    # Combine into a color image.
    color_maps = tf.concat([red_channel, green_channel, blue_channel], axis=3)

    # Scale from 0-255 so they can be displayed easily.
    color_maps = color_maps / max_color_threshold * tf.constant(255.0)
    color_maps = tf.clip_by_value(color_maps, 0.0, 255.0)
    return tf.cast(color_maps, tf.uint8)


def visualize_heat_maps(
    *, images: tf.Tensor, features: tf.Tensor, max_color_threshold: float = 1.0
) -> tf.Tensor:
    """
    Generates a nice visualization for density maps or feature maps that
    overlays the map on the actual image.

    Args:
        images: The raw input images, as a 4D tensor.
        features: The corresponding feature or density maps, as a 4D tensor.
        max_color_threshold: Threshold to use for colorization.
            Any pixel with this value or more will show up as the hottest color.

    Returns:
        A 4D tensor containing heatmap visualizations.

    """
    images = tf.convert_to_tensor(images)
    features = tf.convert_to_tensor(features)

    # Get the colorized density maps.
    colored_maps = _colorize_heat_maps(
        features, max_color_threshold=max_color_threshold
    )

    # Scale to the same size as the input.
    input_size = tf.shape(images)[1:3]
    colored_maps = tf.image.resize(colored_maps, size=input_size)
    colored_maps = tf.cast(colored_maps, tf.uint8)

    # Overlay the density maps on the images.
    images = tf.cast(images, tf.uint8)
    divisor = tf.constant(2, dtype=tf.uint8)
    return images // divisor + colored_maps // divisor
