"""
Nodes for the `annotation_reupload` pipeline.
"""


import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger


def combine_inputs_and_targets(
    patch_dataset: tf.data.Dataset,
) -> tf.data.Dataset:
    """
    For ease of preprocessing, combines the separate inputs and targets
    dictionaries in a dataset into a single dictionary.

    Args:
        patch_dataset: The dataset containing annotated patches.

    Returns:
        The same dataset, with input and target elements combined into one.

    """
    return patch_dataset.map(lambda i, t: {**i, **t})


def annotations_to_dataframe(patch_dataset: tf.data.Dataset) -> pd.DataFrame:
    """
    Creates a `Dataframe` containing the annotation data from a dataset.

    Args:
        patch_dataset: The dataset of patch images. It should already have been
            augmented with saved image paths.

    Returns:
        A `DataFrame` containing the path to the file where each patch is
        stored as well as the ground-truth class for that patch.

    """
    # Iterate one-by-one to make the logic simpler.
    patch_dataset = patch_dataset.unbatch()

    patch_paths = []
    patch_classes = []
    logger.info("Extracting patch and annotation data...")
    for example in patch_dataset:
        patch_path = example["path"].numpy()
        patch_path = patch_path.decode("utf8")
        patch_paths.append(patch_path)

        patch_class = example["discrete_count"].numpy()
        patch_class = patch_class.squeeze().astype(np.int32)
        patch_classes.append(patch_class)

    return pd.DataFrame(data={"paths": patch_paths, "classes": patch_classes})
