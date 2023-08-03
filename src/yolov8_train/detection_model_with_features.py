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

    @classmethod
    def from_detection_model(
        cls, model: DetectionModel
    ) -> "DetectionModelWithFeatures":
        """
        Builds a new instance based on a normal `DetectionModel`.

        Args:
            model: The `DetectionModel`.

        Returns:
            An equivalent `DetectionModelWithFeatures`.

        """
        new_model = DetectionModelWithFeatures()

        # Copy the underlying data.
        new_model.yaml = model.yaml
        new_model.names = model.names
        new_model.inplace = model.inplace
        new_model.model = deepcopy(model.model)

        return new_model

    def _predict_once_with_features(
        self, x: Tensor, profile: bool = False, visualize: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """
        Same as the superclass version, but returns both the detector outputs
        and the extracted features.

        Args:
            x: The inputs to predict on.
            profile: Whether to profile computation time.
            visualize: Whether to visualize feature maps.

        Returns:
            The detection output, and the feature output.

        """
        y, dt = [], []  # outputs
        features = None
        for layer_i, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)

            x = m(x)  # run
            if layer_i == self.__feature_layer:
                # Save the features.
                features = x

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x, features
