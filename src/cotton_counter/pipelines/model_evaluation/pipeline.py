"""
Pipeline that handles evaluating a trained model.
"""


from kedro.pipeline import Pipeline, node

from .nodes import evaluate_model, make_example_density_map


def create_pipeline(**kwargs):
    """
    Defines a pipeline that evaluates a trained model.

    Args:
        kwargs: Ignored by this function.
    """
    # Common parameters shared by evaluation nodes.
    eval_params = dict(
        model="trained_model", classify_counts="params:classify_counts"
    )

    return Pipeline(
        [
            # Evaluate model on all datasets.
            node(
                evaluate_model,
                dict(eval_data="training_data", **eval_params),
                "model_report_train",
            ),
            node(
                evaluate_model,
                dict(eval_data="testing_data", **eval_params),
                "model_report_test",
            ),
            node(
                evaluate_model,
                dict(eval_data="validation_data", **eval_params),
                "model_report_validate",
            ),
            # Create an example density map.
            node(
                make_example_density_map,
                dict(
                    model="trained_model",
                    eval_data="validation_data_no_patches",
                    patch_scale="params:eval_patch_scale",
                    patch_stride="params:eval_patch_stride",
                    batch_size="params:batch_size",
                ),
                "example_density_map",
            ),
        ]
    )
