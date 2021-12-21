"""
Custom Keras layers used by this pipeline.
"""


from kedro.extras.datasets.tensorflow import TensorFlowModelDataset

from .dense import DenseBlock, TransitionLayer
from .mlp_conv import MlpConv
from .ws_count import CombinedBceLoss, CombinedFocalLoss

CUSTOM_OBJECTS = {
    "MlpConv": MlpConv,
    "DenseBlock": DenseBlock,
    "TransitionLayer": TransitionLayer,
    "CombinedBceLoss": CombinedBceLoss,
    "CombinedFocalLoss": CombinedFocalLoss,
}

# Make sure that Kedro is aware of custom layers.
TensorFlowModelDataset.DEFAULT_LOAD_ARGS["custom_objects"] = CUSTOM_OBJECTS
