"""
Pipeline for generating TFRecords from tag-annotated data.
"""


from kedro.pipeline import Pipeline, node

from .nodes import AnnotationFilter, generate_tf_records


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
            # First generate dataset of positive examples.
            node(
                lambda **kw: generate_tf_records(
                    keep_examples=AnnotationFilter.KEEP_POSITIVE, **kw
                ),
                dict(
                    flower_label_name="params:flower_label_name",
                    task_1="cotton_patches_1",
                ),
                "tfrecord_tagged_patches_positive",
            ),
            # Now generate negative examples.
            node(
                lambda **kw: generate_tf_records(
                    keep_examples=AnnotationFilter.KEEP_NEGATIVE, **kw
                ),
                dict(
                    flower_label_name="params:flower_label_name",
                    task_1="cotton_patches_1",
                ),
                "tfrecord_tagged_patches_negative",
            ),
        ]
    )
