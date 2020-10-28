"""
Pipeline for generating TFRecords from tag-annotated data.
"""


from kedro.pipeline import Pipeline, node

from .nodes import generate_tf_records


def create_pipeline(**kwargs):
    """
    Creates the pipeline.

    Args:
        **kwargs: Ignored.

    Returns:
        The pipeline that it created.

    """
    return Pipeline(
        [
            node(
                generate_tf_records,
                dict(
                    flower_label_name="params:flower_label_name",
                    task_1="cotton_patches_1",
                ),
                "tfrecord_tagged_patches",
            )
        ]
    )
