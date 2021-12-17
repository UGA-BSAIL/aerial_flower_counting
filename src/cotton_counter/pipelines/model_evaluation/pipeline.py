"""
Pipeline that handles evaluating a trained model.
"""


from kedro.pipeline import Pipeline, node

from .nodes import (
    calculate_roc_points,
    estimate_counting_accuracy,
    evaluate_model,
    make_accuracy_report,
    make_counting_histogram,
    make_example_density_maps,
    plot_roc_curve,
)


def create_pipeline(**kwargs):
    """
    Defines a pipeline that evaluates a trained model.

    Args:
        kwargs: Ignored by this function.
    """
    # Common parameters shared by evaluation nodes.
    eval_params = dict(model="trained_model")
    # Common parameters used by nodes that do inference on full images.
    inference_params_valid = dict(
        model="trained_model",
        eval_data="validation_data",
        batch_size="params:eval_batch_size",
        patch_scale="params:patch_scale",
    )
    inference_params_test_alt = inference_params_valid.copy()
    inference_params_test_alt["eval_data"] = "testing_data_alternate"

    return Pipeline(
        [
            # Evaluate model on all datasets.
            node(
                evaluate_model,
                dict(eval_data="testing_data", **eval_params),
                "model_report_test",
            ),
            node(
                evaluate_model,
                dict(eval_data="testing_data_alternate", **eval_params),
                "model_report_test_alt",
            ),
            node(
                evaluate_model,
                dict(eval_data="validation_data", **eval_params),
                "model_report_validate",
            ),
            # Calculate overall count accuracy and write reports.
            node(
                estimate_counting_accuracy,
                inference_params_valid,
                ["count_gt_valid", "count_pred_valid"],
                tags="counting_accuracy",
            ),
            node(
                estimate_counting_accuracy,
                inference_params_test_alt,
                ["count_gt_test_alt", "count_pred_test_alt"],
                tags="counting_accuracy",
            ),
            node(
                make_accuracy_report,
                dict(y_true="count_gt_valid", y_pred="count_pred_valid"),
                "count_error_report_valid",
                tags="counting_accuracy",
            ),
            node(
                make_accuracy_report,
                dict(y_true="count_gt_test_alt", y_pred="count_pred_test_alt"),
                "count_error_report_test_alt",
                tags="counting_accuracy",
            ),
            node(
                make_counting_histogram,
                dict(y_true="count_gt_valid", y_pred="count_pred_valid"),
                "count_histogram_valid",
                tags="counting_accuracy",
            ),
            node(
                make_counting_histogram,
                dict(y_true="count_gt_test_alt", y_pred="count_pred_test_alt"),
                "count_histogram_test_alt",
                tags="counting_accuracy",
            ),
            # Create an ROC curve.
            node(
                calculate_roc_points,
                dict(eval_data="validation_data", **eval_params),
                "validation_roc_data",
                tags="counting_accuracy",
            ),
            node(
                calculate_roc_points,
                dict(eval_data="testing_data_alternate", **eval_params),
                "testing_alt_roc_data",
                tags="counting_accuracy",
            ),
            node(
                plot_roc_curve,
                "validation_roc_data",
                "validation_roc_plot",
                tags="counting_accuracy",
            ),
            node(
                plot_roc_curve,
                "testing_alt_roc_data",
                "testing_alt_roc_plot",
                tags="counting_accuracy",
            ),
        ]
    )
