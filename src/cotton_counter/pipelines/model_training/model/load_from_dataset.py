"""
Extracts the information that we need to train the model from a raw
Tensorflow `Dataset`.
"""


import functools
from typing import Iterable, Tuple

import tensorflow as tf

from ..type_helpers import Vector2I
from .density_maps import make_density_maps
from .records import Annotations, ModelInput

_FEATURE_DESCRIPTION = {
    "image": tf.io.FixedLenFeature([1], tf.dtypes.string),
    "frame_numbers": tf.io.RaggedFeature(tf.dtypes.int64),
    "annotation_x": tf.io.RaggedFeature(tf.dtypes.float32),
    "annotation_y": tf.io.RaggedFeature(tf.dtypes.float32),
}
"""
Descriptions of the features found in our dataset.
"""


def _decode_jpeg(jpeg_batch: tf.Tensor) -> tf.Tensor:
    """
    Decodes JPEG images from a feature dictionary.

    Args:
        jpeg_batch: The batch of JPEG images.

    Returns:
        The 4D batch of decoded images.

    """
    # This is going to have a batch dimension, so we need to map it.
    return tf.map_fn(
        lambda j: tf.io.decode_jpeg(j[0]),
        jpeg_batch,
        fn_output_signature=tf.dtypes.uint8,
    )


def _parse_examples(
    serialized: tf.Tensor, *, map_shape: Vector2I, sigma: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Deserializes a set of serialized examples.

    Args:
        serialized: The tensor of serialized examples to parse.
        map_shape: The shape of the output density maps to generate, in the
            form (height, width). The samples dimension will be inferred.
        sigma:
            The standard deviation in pixels to use for the applied gaussian
            filter.

    Returns:
        The images from the batch as well as the generated density maps.

    """
    feature_dict = tf.io.parse_example(serialized, _FEATURE_DESCRIPTION)

    # Decode the JPEG images.
    images = _decode_jpeg(feature_dict["image"])

    # Create density maps.
    annotations = Annotations(
        frame_numbers=feature_dict["frame_numbers"],
        x_values=feature_dict["annotation_x"],
        y_values=feature_dict["annotation_y"],
    )
    density_maps = make_density_maps(
        annotations, map_shape=map_shape, sigma=sigma
    )

    return images, density_maps


def extract_model_input(
    raw_dataset: tf.data.Dataset,
    *,
    map_shape: Vector2I,
    sigma: int,
    batch_size: int = 32,
    num_prefetch_batches: int = 5,
) -> Iterable[ModelInput]:
    """
    Deserializes raw data from a `Dataset`, and coerces it into the form
    used by the model.

    Args:
        raw_dataset: The raw dataset, containing serialized data.
        map_shape: The shape of the output density maps to generate, in the
            form (height, width). The samples dimension will be inferred.
        sigma:
            The standard deviation in pixels to use for the applied gaussian
            filter.
        batch_size: The size of the batches that we generate.
        num_prefetch_batches: The number of batches to prefetch into memory.
            Increasing this can increase performance at the expense of memory
            usage.

    Yields:
        `ModelInput` structures containing the loaded batch data.

    """
    # Batch the data before deserializing for efficiency.
    batched = raw_dataset.batch(batch_size)
    # Deserialize and wrangle it.
    feature_dataset = batched.map(
        functools.partial(_parse_examples, map_shape=map_shape, sigma=sigma)
    )

    # Prefetch the batches.
    feature_dataset = feature_dataset.prefetch(num_prefetch_batches)

    for images, density_maps in feature_dataset:
        yield ModelInput(images=images, density_maps=density_maps)
