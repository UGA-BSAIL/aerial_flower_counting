"""
Contains data to use for testing.
"""


from pathlib import Path

_PACKAGE_PATH = Path(__file__).resolve().parent

IMAGE_SHAPE = (3456, 4608)
"""
The height and width of the images contained in this dataset.
"""

TEST_DATASET_PATH = _PACKAGE_PATH / "testing_dataset.tfrecord"
"""
Path to an example TFRecord dataset that we can use for testing.
"""

TEST_IMAGE_PATH = _PACKAGE_PATH / "test_image.png"
"""
Path to an example image that we can use for testing.
"""
