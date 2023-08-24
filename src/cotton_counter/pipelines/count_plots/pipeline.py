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
    add_plot_index,
    clean_empty_plots,
    clean_genotypes,
    clean_ground_truth,
    clean_height_ground_truth,
    collect_plot_heights,
    compute_counts,
    compute_cumulative_counts,
    compute_flowering_duration,
    compute_flowering_peak,
    compute_flowering_ramps,
    compute_flowering_start_end,
    compute_heights,
    create_height_table,
    create_metric_table,
    create_per_plot_table,
    detect_flowers,
    merge_ground_truth,
    merge_height_ground_truth,
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
    plot_height_comparison,
    plot_height_dist,
    plot_height_ground_truth_regression,
    plot_mean_flowering_curve,
    plot_peak_flowering_comparison,
    plot_peak_flowering_dist,
    segment_flowers,
    load_sam_model,
    merge_dicts,
    compute_flower_sizes,
    plot_flower_size_comparison,
    draw_qualitative_results,
)
from ..common import collect_session_results, filter_low_confidence

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


def _create_session_detection_pipeline(session: str) -> Tuple[Pipeline, str]:
    """
    Creates a pipeline that can be used to detect one session's worth of data.

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


def _create_session_segmentation_pipeline(
    session: str,
) -> Tuple[Pipeline, str]:
    """
    Creates a pipeline that can be used to segment one session's worth of data.

    Args:
        session: The session to make a pipeline for.

    Returns:
        The pipeline that it created, as well as the name of the output node
            for the detections.

    """
    segment_flowers_session = partial(segment_flowers, session_name=session)

    output_node = f"masks_{session}"
    return (
        Pipeline(
            [
                node(
                    segment_flowers_session,
                    dict(
                        images=f"plots_{session}",
                        detections=f"detection_results",
                        segmentor="sam_model",
                    ),
                    output_node,
                )
            ]
        ),
        output_node,
    )


def create_pipeline(**kwargs):
    pipeline = Pipeline([])

    # Create session-specific pipelines for detection.
    session_detection_nodes = []
    for session in SESSIONS:
        session_pipeline, output_node = _create_session_detection_pipeline(
            session
        )
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
                    detection_results="detection_results",
                    min_confidence="params:min_confidence",
                ),
                "filtered_detection_results",
                name="filter_results",
            ),
            node(
                load_sam_model,
                dict(
                    model_type="params:sam_model_type",
                    weights_file="params:sam_weights_file",
                ),
                "sam_model",
            ),
        ]
    )

    # Perform segmentation.
    session_segmentation_nodes = []
    for session in SESSIONS:
        session_pipeline, output_node = _create_session_segmentation_pipeline(
            session
        )
        pipeline += session_pipeline
        session_segmentation_nodes.append(output_node)

    pipeline += Pipeline(
        [
            # Merge segmentations.
            node(merge_dicts, session_segmentation_nodes, "mask_results",),
            # Compute counts.
            node(
                FieldConfig.from_dict,
                "j1_field_config_params",
                "j1_field_config",
            ),
            node(
                FieldConfig.from_dict,
                "j2_field_config_params",
                "j2_field_config",
            ),
            node(
                compute_counts,
                "filtered_detection_results",
                "counting_results_no_dap_no_plot",
            ),
            node(
                add_plot_index,
                dict(
                    plot_data="counting_results_no_dap_no_plot",
                    field_config="j2_field_config",
                ),
                "counting_results_no_dap",
            ),
            node(
                add_dap_counting,
                dict(
                    counting_results="counting_results_no_dap",
                    field_planted_date="params:field_planted_date",
                ),
                "counting_results_with_empty",
            ),
            node(
                compute_cumulative_counts,
                "counting_results_with_empty",
                "cumulative_counts_with_empty",
            ),
            # Compute flower sizes.
            node(
                compute_flower_sizes,
                dict(
                    detection_results="filtered_detection_results",
                    flower_masks="mask_results",
                    gsds="params:session_gsds",
                ),
                "flower_sizes_no_dap_no_plot",
            ),
            node(
                add_plot_index,
                dict(
                    plot_data="flower_sizes_no_dap_no_plot",
                    field_config="j2_field_config",
                ),
                "flower_sizes",
            ),
            # Compute heights.
            # node(
            #     compute_heights, "plots_dem_j1", "detection_plot_heights_j1",
            # ),
            # node(
            #     add_plot_index,
            #     dict(
            #         plot_data="detection_plot_heights_j1",
            #         field_config="j1_field_config",
            #     ),
            #     "plot_heights_j1_with_empty",
            # ),
            # node(
            #     compute_heights, "plots_dem_j2", "detection_plot_heights_j2",
            # ),
            # node(
            #     add_plot_index,
            #     dict(
            #         plot_data="detection_plot_heights_j2",
            #         field_config="j2_field_config",
            #     ),
            #     "plot_heights_j2_with_empty",
            # ),
            # node(
            #     collect_plot_heights,
            #     [f"plot_heights_{field}_with_empty" for field in ("j1", "j2")],
            #     "plot_heights_with_empty",
            # ),
            # Clean data from empty plots.
            node(
                clean_empty_plots,
                dict(
                    plot_df="counting_results_with_empty",
                    empty_plots="empty_plots_j2",
                    field_config="j2_field_config",
                ),
                "counting_results",
            ),
            node(
                clean_empty_plots,
                dict(
                    plot_df="cumulative_counts_with_empty",
                    empty_plots="empty_plots_j2",
                    field_config="j2_field_config",
                ),
                "cumulative_counts",
            ),
            # node(
            #     clean_empty_plots,
            #     dict(
            #         plot_df="plot_heights_with_empty",
            #         empty_plots="empty_plots_j2",
            #         field_config="j2_field_config",
            #     ),
            #     "plot_heights",
            # ),
            # Create the output count table.
            node(
                create_per_plot_table,
                "counting_results",
                "human_readable_counts",
            ),
            # Create the output height table.
            # node(
            #     create_height_table, "plot_heights", "human_readable_heights",
            # ),
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
            # Save the combined results.
            node(
                create_metric_table,
                dict(
                    peak_flowering_times="flowering_peaks",
                    flowering_starts="flowering_starts",
                    flowering_ends="flowering_ends",
                    flowering_durations="flowering_durations",
                    flowering_slopes="flowering_slopes",
                    genotypes="cleaned_genotypes",
                ),
                "human_readable_metrics",
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
            # node(
            #     plot_height_dist,
            #     dict(
            #         plot_heights="plot_heights", genotypes="cleaned_genotypes",
            #     ),
            #     "plot_height_histogram",
            # ),
            # node(
            #     plot_height_comparison,
            #     dict(
            #         plot_heights="plot_heights", genotypes="cleaned_genotypes",
            #     ),
            #     "plot_height_comparison",
            # ),
            node(
                plot_flower_size_comparison,
                dict(
                    flower_sizes="flower_sizes", genotypes="cleaned_genotypes"
                ),
                "flower_size_comparison",
            ),
            # node(
            #     plot_flowering_curves,
            #     dict(
            #         cumulative_counts="cumulative_counts",
            #         genotypes="cleaned_genotypes",
            #     ),
            #     "flowering_curves",
            # ),
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
            # Compare with ground-truth height data.
            node(
                clean_height_ground_truth,
                "ground_truth_height_spreadsheet",
                "clean_gt_heights",
            ),
            # node(
            #     merge_height_ground_truth,
            #     dict(
            #         plot_heights="plot_heights",
            #         ground_truth_heights="clean_gt_heights",
            #     ),
            #     "heights_with_gt",
            # ),
            # node(
            #     plot_height_ground_truth_regression,
            #     "heights_with_gt",
            #     "height_ground_truth_regression_plot",
            # ),
            # Draw qualitative results.
            node(
                draw_qualitative_results,
                dict(
                    qualitative_examples="qualitative_examples",
                    detection_results="filtered_detection_results",
                    flower_masks="mask_results",
                ),
                "qualitative_example_detections",
            ),
        ]
    )

    return pipeline
