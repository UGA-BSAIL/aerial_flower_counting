"""
High-level Python API and utilities for [CVAT](https://github.com/opencv/cvat).
"""

# Also provide some aliases for common types from Swagger.
from swagger_client import Label, LabeledImage, LabeledShape, LabeledTrack

from .dataset.job import Job
from .dataset.task import Task
from .kedro.cvat_data_set import CvatDataSet
