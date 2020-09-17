"""
Implementation of additional image filtering.
"""

import tensorflow as tf


def gaussian_blur(image: tf.Tensor, kernel_size: int = 11, sigma: float = 5):
    """
    Gaussian blur function borrowed from
    https://gist.github.com/blzq/c87d42f45a8c5a53f5b393e27b1f5319

    Yes, I'm *aware* that this exists as part of TF addons, but I can't install
    that because my old TF version forces the use of an old TF addons version,
    and the latter has a broken dependency specification which Poetry chokes on.

    Sometimes I wish I had majored in bio.

    Args:
        image: The 3D image or 4D image batch that we want to blur.
        kernel_size: The size of the kernel to use.
        sigma: The standard deviation to use for the blur.

    Returns:
        The blurred image or images.

    """

    def gauss_kernel(channels: int, _kernel_size: int, _sigma: float):
        ax = tf.range(-_kernel_size // 2 + 1.0, _kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * _sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])

        return kernel

    gaussian_kernel = gauss_kernel(tf.shape(image)[-1], kernel_size, sigma)
    gaussian_kernel = gaussian_kernel[..., tf.newaxis]

    return tf.nn.depthwise_conv2d(
        image,
        gaussian_kernel,
        [1, 1, 1, 1],
        padding="SAME",
        data_format="NHWC",
    )
