"""
Pipeline for auto-counting.
"""


from functools import partial
from typing import Tuple

from kedro.pipeline import Pipeline, node

from .nodes import (
    FieldConfig,
    add_dap_counting,
    add_dap_ground_truth,
    clean_genotypes,
    clean_ground_truth,
    collect_session_results,
    compute_counts,
    compute_cumulative_counts,
    compute_flowering_duration,
    compute_flowering_peak,
    compute_flowering_ramps,
    compute_flowering_start_end,
    create_per_plot_table,
    detect_flowers,
    filter_low_confidence,
    merge_ground_truth,
    plot_flowering_curves,
    plot_flowering_duration_comparison,
    plot_flowering_duration_dist,
    plot_flowering_end_comparison,
    plot_flowering_end_dist,
    plot_flowering_slope_comparison,
    plot_flowering_slope_dist,
    plot_flowering_start_comparison,
    plot_flowering_start_dist,
    plot_ground_truth_regression,
    plot_ground_truth_vs_predicted,
    plot_mean_flowering_curve,
    plot_peak_flowering_comparison,
    plot_peak_flowering_dist,
    find_empty_plots,
)

SESSIONS = {
    "2021-08-09",
    "2021-08-13",
    "2021-08-16",
    "2021-08-20",
    "2021-08-27",
    "2021-08-30",
    "2021-09-03",
    "2021-09-06",
    "2021-09-10",
    "2021-09-13",
    "2021-09-17",
    "2021-09-24",
    "2021-09-27",
    "2021-10-01",
    "2021-10-08",
    "2021-10-11",
    "2021-10-15",
    "2021-10-18",
    "2021-10-22",
    "2021-10-25",
    "2021-10-29",
    "2021-11-01",
    "2021-11-05",
}
"""
The set of all the sessions that we want to process.
"""


def _create_session_pipeline(session: str) -> Tuple[Pipeline, str]:
    """
    Creates a pipeline that can be used to process one session's worth of data.

    Args:
        session: The session to make a pipeline for.

    Returns:
        The pipeline that it created, as well as the name of the output node
            for the detections.

    """
    detect_flowers_session = partial(detect_flowers, session_name=session)

    output_node = f"detections_{session}"
    return (
        Pipeline(
            [
                node(
                    detect_flowers_session,
                    dict(
                        images=f"plots_{session}",
                        weights_file="params:model_weights_file",
                        image_size="params:inference_size",
                    ),
                    output_node,
                )
            ]
        ),
        output_node,
    )


def create_pipeline(**kwargs):
    pipeline = Pipeline([])

    # Create session-specific pipelines.
    session_detection_nodes = []
    for session in SESSIONS:
        session_pipeline, output_node = _create_session_pipeline(session)
        pipeline += session_pipeline
        session_detection_nodes.append(output_node)

    # Create combined pipeline.
    pipeline += Pipeline(
        [
            # Combine the session detections into a single table.
            node(
                collect_session_results,
                session_detection_nodes,
                "detection_results",
            ),
            # Filter low confidence detections.
            node(
                filter_low_confidence,
                dict(
                    counting_results="detection_results",
                    min_confidence="params:min_confidence",
                ),
                "filtered_detection_results",
                name="filter_results",
            ),
            # Compute counts.
            node(
                FieldConfig,
                dict(
                    num_plots="params:field_num_plots",
                    first_row_num="params:field_first_row_num",
                    first_plot_num="params:field_first_plot_num",
                    empty_rows="params:field_empty_rows",
                ),
                "field_config",
                name="field_config",
            ),
            node(
                compute_counts,
                dict(
                    detection_results="filtered_detection_results",
                    field_config="field_config",
                ),
                "counting_results_no_dap",
            ),
            node(
                add_dap_counting,
                dict(
                    counting_results="counting_results_no_dap",
                    field_planted_date="params:field_planted_date",
                ),
                "counting_results",
            ),
            node(
                compute_cumulative_counts,
                "counting_results",
                "cumulative_counts",
            ),
            node(
                find_empty_plots,
                dict(
                    cumulative_counts="cumulative_counts",
                    num_empty_plots="params:num_empty_plots",
                ),
                "empty_plots",
            ),
            # Create the output count table.
            node(
                create_per_plot_table,
                dict(
                    counting_results="counting_results",
                    field_config="field_config",
                ),
                "human_readable_counts",
            ),
            # Perform genotype analysis.
            node(
                clean_genotypes,
                "genotype_spreadsheet",
                "cleaned_genotypes",
                name="clean_genotypes",
            ),
            node(
                compute_flowering_peak, "counting_results", "flowering_peaks"
            ),
            node(
                compute_flowering_start_end,
                dict(
                    counting_results="counting_results",
                    start_threshold="params:flower_start_threshold",
                    end_threshold="params:flower_end_threshold",
                ),
                ["flowering_starts", "flowering_ends"],
            ),
            node(
                compute_flowering_duration,
                dict(
                    flowering_starts="flowering_starts",
                    flowering_ends="flowering_ends",
                ),
                "flowering_durations",
            ),
            node(
                compute_flowering_ramps,
                dict(
                    peak_flowering_times="flowering_peaks",
                    flowering_start_times="flowering_starts",
                    cumulative_counts="cumulative_counts",
                ),
                "flowering_slopes",
            ),
            # Plot the results.
            node(
                plot_peak_flowering_dist,
                dict(
                    peak_flowering_times="flowering_peaks",
                    genotypes="cleaned_genotypes",
                ),
                "peak_flowering_histogram",
            ),
            node(
                plot_peak_flowering_comparison,
                dict(
                    peak_flowering_times="flowering_peaks",
                    genotypes="cleaned_genotypes",
                ),
                "peak_flowering_comparison",
            ),
            node(
                plot_flowering_start_dist,
                dict(
                    flowering_start_times="flowering_starts",
                    genotypes="cleaned_genotypes",
                ),
                "flowering_start_histogram",
            ),
            node(
                plot_flowering_start_comparison,
                dict(
                    flowering_start_times="flowering_starts",
                    genotypes="cleaned_genotypes",
                ),
                "flowering_start_comparison",
            ),
            node(
                plot_flowering_end_dist,
                dict(
                    flowering_end_times="flowering_ends",
                    genotypes="cleaned_genotypes",
                ),
                "flowering_end_histogram",
            ),
            node(
                plot_flowering_end_comparison,
                dict(
                    flowering_end_times="flowering_ends",
                    genotypes="cleaned_genotypes",
                ),
                "flowering_end_comparison",
            ),
            node(
                plot_flowering_duration_dist,
                dict(
                    flowering_durations="flowering_durations",
                    genotypes="cleaned_genotypes",
                ),
                "flowering_duration_histogram",
            ),
            node(
                plot_flowering_duration_comparison,
                dict(
                    flowering_durations="flowering_durations",
                    genotypes="cleaned_genotypes",
                ),
                "flowering_duration_comparison",
            ),
            node(
                plot_flowering_slope_dist,
                dict(
                    flowering_slopes="flowering_slopes",
                    genotypes="cleaned_genotypes",
                ),
                "flowering_slope_histogram",
            ),
            node(
                plot_flowering_slope_comparison,
                dict(
                    flowering_slopes="flowering_slopes",
                    genotypes="cleaned_genotypes",
                ),
                "flowering_slope_comparison",
            ),
            node(
                plot_flowering_curves,
                dict(
                    cumulative_counts="cumulative_counts",
                    genotypes="cleaned_genotypes",
                ),
                "flowering_curves",
            ),
            node(
                plot_mean_flowering_curve,
                dict(
                    cumulative_counts="cumulative_counts",
                    genotypes="cleaned_genotypes",
                ),
                "mean_flowering_curve",
            ),
            # Compare with ground-truth.
            node(
                clean_ground_truth,
                "ground_truth_spreadsheet",
                "cleaned_ground_truth_no_dap",
                name="clean_ground_truth",
            ),
            node(
                add_dap_ground_truth,
                dict(
                    ground_truth="cleaned_ground_truth_no_dap",
                    field_planted_date="params:field_planted_date",
                ),
                "cleaned_ground_truth",
            ),
            node(
                merge_ground_truth,
                dict(
                    counting_results="counting_results",
                    ground_truth="cleaned_ground_truth",
                ),
                "counts_with_gt",
            ),
            node(
                plot_ground_truth_regression,
                "counts_with_gt",
                "ground_truth_regression_plot",
            ),
            node(
                plot_ground_truth_vs_predicted,
                dict(
                    counts_with_gt="counts_with_gt",
                    genotypes="cleaned_genotypes",
                ),
                "ground_truth_vs_predicted",
            ),
        ]
    )

    return pipeline
