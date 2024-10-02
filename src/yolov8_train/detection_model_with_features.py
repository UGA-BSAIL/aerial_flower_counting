from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.plotting import feature_visualization
from copy import deepcopy
from torch import Tensor
from typing import Tuple, Any


class DetectionModelWithFeatures(DetectionModel):
    """
    Special detection model that returns the raw features as well.
    """

    def __init__(self, *args: Any, feature_layer: int = 5, **kwargs: Any):
        """
        Args:
            *args: Forwarded to superclass.
            feature_layer: Index of the layer to extract features from.
            **kwargs: Forwarded to superclass.

        """
        self.__feature_layer = feature_layer

        super().__init__(*args, **kwargs)

        # This is somewhat hacky, but during initialization, it expects
        # `_predict_once()` to return just one Tensor. To get around this,
        # we don't swap it out until now.
        self._predict_once = self._predict_once_with_features

    def predict(self, *args: Any, **kwargs: Any):
        """
        Calls the superclass's `predict()` method, but defaults to also
        returning feature embeddings.

        Args:
            *args: Forwarded to superclass.
            **kwargs: Forwarded to superclass.

        """
        super().predict(*args, **kwargs, embed=self.__feature_layer)
