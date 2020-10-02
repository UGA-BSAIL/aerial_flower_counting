"""
Custom Keras layers used by this pipeline.
"""


from kedro.extras.datasets.tensorflow import TensorFlowModelDataset

from .mlp_conv import MlpConv

# Make sure that Kedro is aware of custom layers.
TensorFlowModelDataset.DEFAULT_LOAD_ARGS["custom_objects"] = {
    "MlpConv": MlpConv
}
