"""
Represents model input and output data in a standard form.
"""


import tensorflow as tf
from pydantic.dataclasses import dataclass

from ..type_helpers import ArbitraryTypesConfig, MaybeRaggedTensor


@dataclass(config=ArbitraryTypesConfig)
class Annotations:
    """
    Represents annotations for a set of images.

    Attributes:
        frame_numbers: The vector of frame numbers in the batch.
        x_values: The vector of x-values for the annotations.
        y_values: The vector of y-values for the annotations.
    """

    frame_numbers: MaybeRaggedTensor
    x_values: MaybeRaggedTensor
    y_values: MaybeRaggedTensor


@dataclass(frozen=True, config=ArbitraryTypesConfig)
class ModelInferenceInput:
    """
    Represents minimal input for the model, such as what is used during
    prediction.

    Attributes:
        images: The batch of input images, in JPEG form.
    """

    images: tf.Tensor


@dataclass(frozen=True, config=ArbitraryTypesConfig)
class ModelInput(ModelInferenceInput):
    """
    Represents the complete training input for the model.

    Attributes:
        density_maps: The generated density maps for this batch.
    """

    density_maps: tf.Tensor
