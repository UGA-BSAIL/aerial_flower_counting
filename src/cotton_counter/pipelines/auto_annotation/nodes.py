"""
Node functions for the `auto_annotation` pipeline.
"""


from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from pycvat import Label, LabeledImage, Task

from ...cvat_utils import get_main_job
from ...model.dataset_io import inputs_from_generator, save_images_to_disk
from ...type_helpers import Vector2I


def unannotated_patch_dataset(
    *,
    cvat_task: Task,
    start_frame_num: int,
    num_frames: int,
    annotation_interval: int = 1,
    **kwargs: Any,
) -> tf.data.Dataset:
    """
    Creates a dataset containing all the unannotated images from CVAT in raw
    JPEG form.

    Args:
        cvat_task: The CVAT task to pull data from.
        start_frame_num: The frame number to start at.
        num_frames: Maximum number of frames to iterate over.
        annotation_interval: Number of frames to skip between each annotated
            one. This can be useful for videos with many similar frames.
        **kwargs: Will be forwarded to `inputs_from_generator`.

    Returns:
        The `Dataset` that it created.

    """

    def iter_unannotated_images() -> Iterable[np.ndarray]:
        """
        Generator that produces all the unannotated images from CVAT.

        Yields:
            Each unannotated image.

        """
        num_frames_produced = 0
        job = get_main_job(cvat_task)

        for i, annotations in enumerate(
            job.iter_annotations(start_at=start_frame_num),
        ):
            if i % annotation_interval != 0:
                # This frame should be skipped.
                continue

            frame_num = i + start_frame_num

            if num_frames_produced >= num_frames:
                # We've produced a sufficient number of frames.
                logger.debug("Done producing frames.")
                break

            if len(annotations) == 0:
                # This image has no annotations.
                num_frames_produced += 1
                yield cvat_task.get_image(frame_num, compressed=True).tobytes()

    # Convert to a Dataset that we can use with the model.
    return inputs_from_generator(
        iter_unannotated_images, extract_jpegs=True, **kwargs
    )


def coerce_patch_shapes(
    *, patches: tf.data.Dataset, desired_shape: Vector2I
) -> tf.data.Dataset:
    """
    Modifies a dataset of patches such that the images have a desired shape.
    It will not change the aspect ratio, so this may involve cropping out the
    edges.

    Args:
        patches: The dataset of raw patches.
        desired_shape: The desired shape for the _patches, in the form
            (height, width)

    Returns:
        A new dataset containing the reshaped _patches.

    """
    desired_height, desired_width = desired_shape
    desired_aspect_ratio = desired_width / desired_height

    def _coerce_patch_shape(_patches: tf.Tensor) -> tf.Tensor:
        """
        Coerces the shape of a single patch.

        Args:
            _patches: The batch of patches to reshape.

        Returns:
            The reshaped patch.

        """
        input_shape = tf.shape(_patches)
        input_height = input_shape[1]
        input_width = input_shape[2]
        input_aspect_ratio = tf.cast(input_width, tf.float32) / tf.cast(
            input_height, tf.float32
        )

        # Calculate what size to rescale to, while maintaining the aspect ratio.
        scaled_size = tf.cond(
            input_aspect_ratio < desired_aspect_ratio,
            # If the desired size is wider that the input, scale to the width.
            true_fn=lambda: (
                tf.cast(desired_width / input_aspect_ratio, tf.int32),
                desired_width,
            ),
            # If the desired size is taller than the input, scale to the height.
            false_fn=lambda: (
                desired_height,
                tf.cast(desired_height * input_aspect_ratio, tf.int32),
            ),
        )

        resized = tf.image.resize(_patches, size=scaled_size, method="bicubic")
        # Output from resize() is always float32. We clip to avoid any color
        # artifacts.
        resized = tf.cast(tf.clip_by_value(resized, 0, 255), tf.uint8)

        # Crop away any over-sized portions of the image.
        return tf.image.resize_with_crop_or_pad(
            resized, desired_height, desired_width
        )

    return patches.map(lambda e: {"image": _coerce_patch_shape(e["image"])})


def save_patches_to_disk(
    *, patches: tf.data.Dataset, batch_size: int
) -> Tuple[tf.data.Dataset, TemporaryDirectory]:
    """
    Saves extracted image patches to a temporary directory.

    Args:
        patches: The dataset containing image patches.
        batch_size: The size of the batches that we want to use.

    Returns:
        A new dataset which contains the same patches, as well as the
        corresponding path to the image file that was created for that patch. It
        also returns the temporary directory object that was created so that
        it can be destroyed at our leisure.

    """
    # Remove the batching on the patches.
    patches_flat = patches.unbatch()

    # Save the data.
    temp_dir = TemporaryDirectory()
    temp_dir_path = Path(temp_dir.name)
    logger.info("Saving image patches to {}.", temp_dir_path)
    patches_with_paths = save_images_to_disk(
        patches_flat, save_dir=temp_dir_path
    )

    # Reinstate batching.
    patches_with_paths = patches_with_paths.batch(batch_size)

    return patches_with_paths, temp_dir


def predict_patches(
    *, model: tf.keras.Model, patch_dataset: tf.data.Dataset
) -> pd.DataFrame:
    """
    Generates predictions for a set of patches using a provided model.

    Args:
        model: The model to use for making predictions.
        patch_dataset: The dataset of patch images to predict.

    Returns:
        A `DataFrame` containing the path to the file where each patch is
        stored as well as the predicted class for that patch.

    """
    # Modify the model so that is passes the image path through to the output.
    path_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="path")
    model_with_path = tf.keras.Model(
        inputs=model.inputs + [path_input],
        # We only care about the discrete count output here.
        outputs=model.outputs + [path_input],
    )

    # Predict on the patches.
    model_output = model_with_path.predict(patch_dataset)

    # Organize the resulting data.
    patch_sigmoid, _, patch_paths = model_output
    patch_classes = patch_sigmoid > 0.5
    patch_classes = patch_classes.squeeze().astype(np.int32)
    patch_paths = patch_paths.squeeze()
    patch_paths = [p.decode("utf8") for p in patch_paths]
    return pd.DataFrame(data={"paths": patch_paths, "classes": patch_classes})


def _update_cvat_annotations(
    *, annotations: pd.DataFrame, cvat_task: Task, label: Label
) -> None:
    """
    Updates the annotations on CVAT based on annotations stored in a
    `DataFrame`.

    Args:
        annotations: The annotations that we want to update CVAT with.
        cvat_task: The CVAT task to update.
        label: The label we add to images to indicate a positive detection.

    """
    # Our model annotates positive classes with 0.
    positive_examples = annotations["classes"] == 0
    # CVAT names things using the file name.
    positive_examples_names = [
        Path(p).name for p in annotations[positive_examples]["paths"]
    ]
    logger.debug(
        "Have {} frames with positive annotations.",
        len(positive_examples_names),
    )

    # Update the annotations.
    job = get_main_job(cvat_task)
    for image_name in positive_examples_names:
        # We can't trust that CVAT will leave images in the same order that
        # we uploaded them, hence the searching by name.
        frame_num = cvat_task.find_image_frame_num(image_name)
        annotation = LabeledImage(
            frame=frame_num, label_id=label.id, group=0, attributes=[]
        )
        job.update_annotations(annotation)
    job.upload()


def upload_patches(*, annotations: pd.DataFrame, cvat_task: Task) -> None:
    """
    Uploads annotated patches to CVAT as a new task.

    Args:
        annotations: The `DataFrame` containing the auto-generated annotations.
        cvat_task: The CVAT task that the data were originally downloaded from.

    """
    # Create a new CVAT task on the same server as the original.
    num_patches = len(annotations)
    task_name = (
        f"Auto Annotation of {num_patches} Images at "
        f"{datetime.now().isoformat()}"
    )
    logger.info("Creating new task: {}", task_name)

    # For these data, we labels for the presence of pollinated and
    # unpollinated flowers.
    unpollinated_label = Label(name="has_flower", attributes=[])
    pollinated_label = Label(name="has_pollinated_flower", attributes=[])

    task = Task.create_new(
        api_client=cvat_task.api,
        name=task_name,
        labels=[unpollinated_label, pollinated_label],
        images=annotations["paths"],
    )
    # Refresh the label definitions so that they're populated with the actual
    # IDs.
    unpollinated_label = task.find_label(unpollinated_label.name)

    _update_cvat_annotations(
        annotations=annotations, cvat_task=task, label=unpollinated_label,
    )
