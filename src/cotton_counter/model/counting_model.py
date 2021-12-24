"""
Implements the model architecture.
"""

from typing import Any, Callable, Iterable, Optional, Tuple

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
from loguru import logger

from src.cotton_counter.type_helpers import Vector2I

from ..model.layers import CombinedFocalLoss, DenseBlock, TransitionLayer
from ..model.patches import extract_standard_patches
from .losses import FocalLoss


def _build_image_input(*, input_size: Vector2I) -> keras.Input:
    """
    Creates the image data input.

    Args:
        input_size: The size of the input images that will be provided,
            in the form (width, height).

    Returns:
        The `Input` that it created.

    """
    # Assume RGB images.
    input_shape = input_size[::-1] + (3,)
    logger.debug("Creating model with input shape {}.", input_shape)
    return keras.Input(shape=input_shape, name="image", dtype="uint8")


def _bn_relu_conv(
    *args: Any, **kwargs: Any
) -> Callable[[tf.Tensor], tf.Tensor]:
    """
    Small helper function that builds a bn-relu-conv block.

    Args:
        *args: Forwarded to `Conv2D()`.
        **kwargs: Forwarded to `Conv2D()`.

    Returns:
        The block, which can be called to apply it to some input, similar to
        a Keras layer.

    """
    conv = layers.Conv2D(*args, **kwargs)
    norm = layers.BatchNormalization()
    relu = layers.Activation("relu")

    def _apply_block(block_input: tf.Tensor) -> tf.Tensor:
        return conv(relu(norm(block_input)))

    return _apply_block


def _build_dense_net_backbone(normalized_input: tf.Tensor) -> tf.Tensor:
    """
    Creates a DenseNet model.

    Args:
        normalized_input: The normalized input images.

    Returns:
        The top model layer.

    """
    # Input convolution layers.
    conv1_1 = _bn_relu_conv(64, 3, padding="same")(normalized_input)
    conv1_2 = _bn_relu_conv(64, 3, padding="same")(conv1_1)
    pool1 = layers.MaxPool2D()(conv1_2)

    # Dense blocks.
    dense1 = DenseBlock(6, growth_rate=12)(pool1)
    transition1 = TransitionLayer()(dense1)

    dense2 = DenseBlock(12, growth_rate=12)(transition1)
    transition2 = TransitionLayer()(dense2)

    dense3 = DenseBlock(12, growth_rate=12)(transition2)
    transition3 = TransitionLayer()(dense3)

    dense4 = DenseBlock(8, growth_rate=12)(transition3)

    return dense4


def _build_le_net_backbone(normalized_input: tf.Tensor) -> tf.Tensor:
    """
    Creates a model similar to the LeNet-based architecture used in the
    TasselNet paper. Note, however, that the input size is much larger.

    Args:
        normalized_input: The normalized input images.

    Returns:
        The top model layer.

    """
    conv1_1 = _bn_relu_conv(16, 3, padding="same")(normalized_input)
    pool1 = layers.MaxPool2D()(conv1_1)

    conv2_1 = _bn_relu_conv(32, 3, padding="same")(pool1)
    pool2 = layers.MaxPool2D()(conv2_1)

    conv3_1 = _bn_relu_conv(64, 8, padding="same")(pool2)
    conv3_2 = _bn_relu_conv(64, 1, padding="same")(conv3_1)

    return conv3_2


def _build_alexnet_backbone(normalized_input: tf.Tensor) -> tf.Tensor:
    """
    Creates a model similar to the AlexNet-based architecture used in the
    TasselNet paper. Note, however, that the input size is much larger.

    Args:
        normalized_input: The normalized input images.

    Returns:
        The top model layer.

    """
    conv1_1 = _bn_relu_conv(16, 3, padding="same")(normalized_input)
    pool1 = layers.MaxPool2D()(conv1_1)

    conv2_1 = _bn_relu_conv(32, 3, padding="same")(pool1)
    pool2 = layers.MaxPool2D()(conv2_1)

    conv3_1 = _bn_relu_conv(64, 3, padding="same")(pool2)
    conv3_2 = _bn_relu_conv(64, 3, padding="same")(conv3_1)
    conv3_3 = _bn_relu_conv(64, 3, padding="same")(conv3_2)
    pool3 = layers.MaxPool2D()(conv3_3)

    conv4_1 = _bn_relu_conv(128, 4, padding="same")(pool3)
    conv4_2 = _bn_relu_conv(128, 1, padding="same")(conv4_1)

    return conv4_2


def _build_vgg_backboune(normalized_input: tf.Tensor) -> tf.Tensor:
    """
    Creates a model similar to the VGG16-based architecture that is used in the
    TasselNet paper. Note, however, that the input size is much larger.

    Args:
        normalized_input: The normalized input images.

    Returns:
        The top model layer.

    """
    conv1_1 = _bn_relu_conv(16, 3, padding="same")(normalized_input)
    conv1_2 = _bn_relu_conv(16, 3, padding="same")(conv1_1)
    pool1 = layers.MaxPool2D()(conv1_2)

    conv2_1 = _bn_relu_conv(32, 3, padding="same")(pool1)
    conv2_2 = _bn_relu_conv(32, 3, padding="same")(conv2_1)
    pool2 = layers.MaxPool2D()(conv2_2)

    conv3_1 = _bn_relu_conv(64, 3, padding="same")(pool2)
    conv3_2 = _bn_relu_conv(64, 3, padding="same")(conv3_1)
    conv3_3 = _bn_relu_conv(64, 3, padding="same")(conv3_2)
    pool3 = layers.MaxPool2D()(conv3_3)

    conv4_1 = _bn_relu_conv(128, 3, padding="same")(pool3)
    conv4_2 = _bn_relu_conv(128, 3, padding="same")(conv4_1)
    conv4_3 = _bn_relu_conv(128, 3, padding="same")(conv4_2)
    pool4 = layers.MaxPool2D()(conv4_3)

    conv4_1 = _bn_relu_conv(256, 3, padding="same")(pool4)
    conv4_2 = _bn_relu_conv(256, 3, padding="same")(conv4_1)
    conv4_3 = _bn_relu_conv(256, 3, padding="same")(conv4_2)
    pool5 = layers.MaxPool2D()(conv4_3)

    conv5_1 = _bn_relu_conv(512, 1, padding="same")(pool5)
    conv5_2 = _bn_relu_conv(512, 1, padding="same")(conv5_1)

    return conv5_2


def _build_model_backbone(*, image_input: keras.Input) -> tf.Tensor:
    """
    Creates the backbone of the model.

    Args:
        image_input: The image input to build the model off of.

    Returns:
        The top model layer.

    """
    # Normalize the images before putting them through the model.
    float_images = tf.cast(image_input, K.floatx())
    normalized = tf.image.per_image_standardization(float_images)

    return _build_dense_net_backbone(normalized)


def _get_sub_patch_features(
    model_top: tf.Tensor, *, num_scales: int = 3
) -> Tuple[tf.Tensor, ...]:
    """
    Takes a set of features, and computes sub-patches of the features at
    multiple scales. Each scale will be 1/4 the size of the previous one.

    Args:
        model_top: The features to compute sub-patches of.
        num_scales: The number of scales to compute sub-patches for. If this
            is one, it will be equivalent to the identity operation.

    Returns:
        The sub-patches at each scale. The first will be the same as the
        input, and each later one will get progressively smaller. The batch
        dimension of scale n corresponds to the flattened sub-patches from
        scale n-1. The output tensor for each scale will have
        the shape (n_batch, patch_height, patch_width, patch_channels),
        with `n_batch` increasing by 4 at every scale.

    """
    assert num_scales >= 1, "Must have at least one scale."

    # Scale to use for patches 1/4 the size of the input.
    sub_patch_scale = 0.5

    # Include a "dummy" output for the initial scale of 1, i.e. exactly the
    # input.
    patches = [model_top]
    # Extract the patches for each scale.
    previous_scale_patches = model_top
    for scale in range(num_scales - 1):
        sub_patches = extract_standard_patches(
            previous_scale_patches,
            patch_scale=sub_patch_scale,
            patch_stride=sub_patch_scale,
            padding="VALID",
        )

        # Group the patches by their corresponding input image.
        patches.append(sub_patches)
        previous_scale_patches = sub_patches

    return tuple(patches)


def _build_neck(
    model_top: tf.Tensor,
    *,
    output_bias: Optional[float] = None,
    name: str = "",
) -> tf.Tensor:
    """
    Adds a neck for either task. Currently, we don't have different
    architectures for the two, so we use the same function for both.

    Args:
        model_top: The top model layer to build the neck on.
        output_bias: Specify an initial bias to use for the output. This can
            be useful for unbalanced datasets.
        name: Name to use when naming the output layer. It will be called
            "activation_maps_{name}".

    Returns:
        A tensor representing the activation maps.

    """
    if output_bias is not None:
        # Use the specified bias.
        logger.debug("Using initial output bias {}.", output_bias)
        output_bias = keras.initializers.Constant(output_bias)

    # Create the layers.
    count_conv_1 = _bn_relu_conv(3, 128, padding="same")(model_top)
    count_conv_2 = _bn_relu_conv(3, 128, padding="same")(count_conv_1)
    return layers.Conv2D(
        1, 1, name=f"activation_maps_{name}", bias_initializer=output_bias
    )(count_conv_2)


def _build_pac_head(model_top: Iterable[tf.Tensor]) -> Iterable[tf.Tensor]:
    """
    Adds the head for classifying patches based on a presence or absence of
    objects.

    Args:
        model_top: The top model layer to build the head on. If provided
            with multiple tensors, it will apply the same layers to each one.

    Returns:
        A tensor representing the presence/absence logits for each input.

    """
    count_pool_1 = layers.GlobalAveragePooling2D()
    count_sigmoid = layers.Activation("sigmoid", name="pac")

    # Apply the layers.
    for top_features in model_top:
        yield count_sigmoid(count_pool_1(top_features))


def _build_count_head_and_neck(
    model_top: Iterable[tf.Tensor],
) -> Iterable[tf.Tensor]:
    """
    Adds the head for counting the number of objects in an input.

    Args:
        model_top: The top model layer to build the head on. If provided
            with multiple tensors, it will apply the separate layers to each
            one.

    Returns:
        A tensor representing the counts for each input.

    """
    # It doesn't really make sense to have a count less than 0.
    count_pool_1 = layers.GlobalAveragePooling2D(name="count")

    # Apply the layers.
    for i, top_features in enumerate(model_top):
        # Build the neck.
        count_neck = _build_neck(top_features, name=f"count_{i}")
        yield count_pool_1(count_neck)


def _compute_scale_consistency_loss(
    count_predictions: Iterable[tf.Tensor],
) -> tf.Tensor:
    """
    Ensures that the predicted counts at each smaller scale sum to the same
    thing as the predicted counts at the larger scale.

    Args:
        count_predictions: The count predictions for each scale. Each element
            should have a shape of (batch_size, 1).

    Returns:
        The total computed MSE loss for every item in the batch. Will be a
        vector with a length of the batch size.

    """
    count_predictions = list(count_predictions)

    batch_size = tf.shape(count_predictions[0])[0]
    total_mse = tf.zeros(tf.expand_dims(batch_size, 0), dtype=tf.float32)

    # Keeps track of the number of patches corresponding to each input image.
    num_patches_per_image = 1
    # Compare the counts in a pair-wise fashion.
    for large_scale, small_scale in zip(
        count_predictions, count_predictions[1:]
    ):
        large_scale = tf.ensure_shape(large_scale, (None, 1))
        small_scale = tf.ensure_shape(small_scale, (None, 1))

        # The smaller scale should have four patches for every image in the
        # larger-scale batch. Add a new dimension to group them by their
        # corresponding image in the larger-scale batch.
        large_scale_grouped_shape = tf.stack(
            (batch_size, num_patches_per_image)
        )
        small_scale_grouped_shape = tf.stack(
            (batch_size, num_patches_per_image * 4)
        )
        large_scale_grouped = tf.reshape(
            large_scale, large_scale_grouped_shape
        )
        small_scale_grouped = tf.reshape(
            small_scale, small_scale_grouped_shape
        )

        # Compute total counts.
        large_scale_total_counts = tf.reduce_sum(large_scale_grouped, axis=1)
        small_scale_total_counts = tf.reduce_sum(small_scale_grouped, axis=1)
        tf.print(
            "large_scale",
            large_scale_total_counts,
            "small_scale",
            small_scale_total_counts,
        )

        # Compute MSE.
        total_mse += tf.keras.losses.mse(
            large_scale_total_counts, small_scale_total_counts
        )

        num_patches_per_image *= 4

    return total_mse


def build_model(
    *,
    input_size: Vector2I,
    num_scales: int,
    output_bias: Optional[float] = None,
    focal_alpha: float = 0.5,
    focal_gamma: float = 1.0,
) -> keras.Model:
    """
    Creates the full SaNet model.

    Args:
        input_size: The size of the input images that will be provided,
            in the form (width, height).
        num_scales: Number of different scales to use for weakly-supervised
            counting.
        output_bias: Specify an initial bias to use for the output. This can
            be useful for unbalanced datasets.
        focal_alpha: Alpha parameter to use for combined focal loss.
        focal_gamma: Gamma parameter to use for combined focal loss.

    Returns:
        The model that it created.

    """
    image_input = _build_image_input(input_size=input_size)
    backbone = _build_model_backbone(image_input=image_input)
    # Create the PAC neck.
    pac_neck = _build_neck(backbone, output_bias=output_bias, name="pac")
    # Compute multi-scale features.
    pac_multi_scale_features = _get_sub_patch_features(
        pac_neck, num_scales=num_scales
    )
    backbone_multi_scale_features = _get_sub_patch_features(
        backbone, num_scales=num_scales
    )

    # Create the heads.
    pac_head = list(_build_pac_head(pac_multi_scale_features))
    count_head = list(
        _build_count_head_and_neck(backbone_multi_scale_features)
    )

    # Compute the combined BCE loss.
    focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    combined_bce_loss = CombinedFocalLoss(
        name="combined_bce", bce_loss=focal_loss
    )(dict(pac_predictions=pac_head, count_predictions=count_head))
    # Compute the scale consistency loss.
    scale_consistency_loss = tf.keras.layers.Lambda(
        _compute_scale_consistency_loss, name="scale_consistency"
    )(count_head)

    model_outputs = dict(
        # Only include outputs for the largest scale.
        has_flower=pac_head[0],
        count=count_head[0],
        # Output the combined BCE and consistency losses also.
        combined_bce_loss=combined_bce_loss,
        scale_consistency_loss=scale_consistency_loss,
    )

    # Create the model.
    model = keras.Model(inputs=image_input, outputs=model_outputs,)

    return model
