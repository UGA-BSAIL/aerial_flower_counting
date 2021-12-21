from pathlib import Path

import tensorflow as tf

from src.cotton_counter.model.counting_model import build_model
from src.cotton_counter.model.layers import CUSTOM_OBJECTS


def test_save_load_smoke(tmp_path: Path) -> None:
    """
    Tests that we can save a model and then load it again.

    Args:
        tmp_path: The directory to use for saving temporary data.

    """
    # Arrange.
    save_path = tmp_path / "test_model.h5"
    # Create the model.
    model = build_model(input_size=(576, 432), num_scales=3)

    # Act and assert.
    model.save(save_path, save_format="h5")
    tf.keras.models.load_model(
        save_path, compile=False, custom_objects=CUSTOM_OBJECTS
    )
