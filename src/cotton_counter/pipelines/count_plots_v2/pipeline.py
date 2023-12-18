"""
Version two of the pipeline for auto-counting.
"""
from functools import partial, reduce
from operator import add
from typing import Tuple

import pandas as pd
from kedro.pipeline import Pipeline, node

from ..camera_utils import CameraConfig
from ..common import (
    BOLL_SESSIONS,
    GT_SESSIONS,
    SESSIONS,
    add_dap_counting,
    add_dap_ground_truth,
    choose_best_counts,
    collect_session_results,
    compute_counts,
    compute_cumulative_counts,
    compute_flowering_duration,
    compute_flowering_peak,
    compute_flowering_ramps,
    compute_flowering_start_end,
    create_metric_table,
    filter_low_confidence,
    merge_ground_truth,
    plot_flowering_duration_comparison,
    plot_flowering_duration_dist,
    plot_flowering_end_comparison,
    plot_flowering_end_dist,
    plot_flowering_slope_comparison,
    plot_flowering_slope_dist,
    plot_flowering_start_comparison,
    plot_flowering_start_dist,
    plot_ground_truth_regression,
    plot_mean_flowering_curve,
    plot_peak_flowering_comparison,
    plot_peak_flowering_dist,
)
from .field_config import FieldConfig
from .nodes import (
    add_plot_index,
    clean_genotypes,
    detect_bolls,
    detect_flowers,
    find_all_outliers,
    find_detections_in_gt_sampling_regions_post_september,
    find_detections_in_gt_sampling_regions_pre_september,
    find_detections_in_plots_post_september,
    find_detections_in_plots_pre_september,
    find_genotypes_to_collect,
    find_image_extents,
    flowers_to_geographic,
    flowers_to_shapefile,
    load_ground_truth,
    plot_ground_truth_vs_predicted,
    prune_duplicate_detections,
)


def _create_session_detection_pipeline(session: str) -> Tuple[Pipeline, str]:
    """
    Creates a pipeline that can be used to detect one session's worth of data.

    Args:
        session: The session to make a pipeline for.

    Returns:
        The pipeline that it created, as well as the name of the output node
            for the detections.

    """
    detection_inputs = dict(
        images=f"images_{session}",
        batch_size="params:prediction_batch_size",
        dry_run="params:dry_run",
        weights_file="params:flower_model_weights_file",
    )
    detect_session = partial(
        detect_flowers,
        session_name=session,
    )
    if session in BOLL_SESSIONS:
        detect_session = partial(
            detect_bolls,
            session_name=session,
        )
        detection_inputs["weights_file"] = "params:boll_model_weights_file"

    output_node = f"detections_{session}"
    return (
        Pipeline(
            [
                # Detect flowers.
                node(
                    detect_session,
                    detection_inputs,
                    f"detections_px_{session}",
                ),
                # Convert to geographic coordinates.
                node(
                    partial(flowers_to_geographic, session_name=session),
                    dict(
                        detections=f"detections_px_{session}",
                        camera_config="camera_config",
                        dem_dataset="dems",
                    ),
                    f"detections_unfiltered_{session}",
                ),
                # Remove duplicate detections.
                node(
                    prune_duplicate_detections,
                    dict(
                        detections=f"detections_unfiltered_{session}",
                        image_extents=f"image_extents_{session}",
                    ),
                    output_node,
                ),
            ]
        ),
        output_node,
    )


def _create_ground_truth_pipeline() -> Pipeline:
    """
    Returns:
        A pipeline that loads all the ground-truth data.

    """
    nodes = []
    session_node_names = []
    for session in GT_SESSIONS:
        load_gt_session = partial(load_ground_truth, session_name=session)
        nodes.append(
            node(
                load_gt_session,
                dict(raw_gt_data=f"gt_{session}"),
                f"gt_{session}_loaded",
            )
        )
        session_node_names.append(f"gt_{session}_loaded")

    nodes.extend(
        [
            # Merge them all together.
            node(lambda *d: pd.concat(d), session_node_names, "gt_no_dap"),
            # Add the DAP value.
            node(
                add_dap_ground_truth,
                dict(
                    ground_truth="gt_no_dap",
                    field_planted_date="params:v2_field_planted_date",
                ),
                "gt_combined",
            ),
        ]
    )

    return Pipeline(nodes)


def _create_image_extents_pipeline() -> Pipeline:
    """
    Returns:
        A pipeline that outputs all the image extents data.

    """
    nodes = []
    session_node_names = []
    for session in SESSIONS:
        image_extents_session = partial(
            find_image_extents, session_name=session
        )
        output_node = f"image_extents_{session}"
        nodes.append(
            node(
                image_extents_session,
                dict(
                    images=f"images_{session}",
                    camera_config="camera_config",
                    dem_dataset="dems",
                ),
                output_node,
            ),
        )
        session_node_names.append(output_node)

    nodes.extend(
        [
            # Merge them all together.
            node(
                lambda *f: reduce(add, f, []),
                session_node_names,
                "image_extents",
            ),
        ]
    )

    return Pipeline(nodes)


def _create_analysis_pipeline() -> Pipeline:
    """
    Returns:
        A pipeline that analyzes computed flower counts.

    """
    return Pipeline(
        [
            # Compute flowering metrics.
            node(
                compute_cumulative_counts,
                "counting_results",
                "cumulative_counts",
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
            # Find outliers.
            node(
                clean_genotypes,
                "genotype_spreadsheet_2023",
                "cleaned_genotypes",
                name="clean_genotypes",
            ),
            node(
                find_all_outliers,
                dict(
                    start="flowering_starts",
                    end="flowering_ends",
                    duration="flowering_durations",
                    peak="flowering_peaks",
                    slope="flowering_slopes",
                    genotypes="cleaned_genotypes",
                ),
                "outliers",
            ),
            # Save the metric table.
            node(
                create_metric_table,
                dict(
                    peak_flowering_times="flowering_peaks",
                    flowering_starts="flowering_starts",
                    flowering_ends="flowering_ends",
                    flowering_durations="flowering_durations",
                    flowering_slopes="flowering_slopes",
                    outliers="outliers",
                    genotypes="cleaned_genotypes",
                ),
                "human_readable_metrics",
            ),
            # Plot flowering metrics.
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
                plot_mean_flowering_curve,
                dict(
                    cumulative_counts="cumulative_counts",
                    genotypes="cleaned_genotypes",
                ),
                "mean_flowering_curve",
            ),
            # Plot the ground truth regression.
            node(
                merge_ground_truth,
                dict(
                    counting_results="counting_results_gt_sample",
                    ground_truth="gt_combined",
                ),
                "counting_results_gt_merged_with_dups",
            ),
            node(
                choose_best_counts,
                "counting_results_gt_merged_with_dups",
                "counting_results_gt_merged",
            ),
            node(
                plot_ground_truth_regression,
                "counting_results_gt_merged",
                "ground_truth_regression_plot",
            ),
            node(
                plot_ground_truth_vs_predicted,
                "counting_results_gt_merged",
                "ground_truth_vs_predicted",
            ),
            # Determine which genotypes should be collected in the field.
            node(
                find_genotypes_to_collect,
                dict(
                    start="flowering_starts",
                    end="flowering_ends",
                    peak="flowering_peaks",
                    slope="flowering_slopes",
                    genotypes="cleaned_genotypes",
                    num_to_select="params:num_genotypes_to_collect",
                    habit_quantiles="params:genotype_habit_quantiles",
                ),
                "genotypes_to_collect",
            ),
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    pipeline = _create_ground_truth_pipeline()
    pipeline += _create_image_extents_pipeline()

    # Create session-specific pipelines for detection.
    session_detection_nodes = []
    for session in SESSIONS:
        (
            session_detection_pipeline,
            output_node,
        ) = _create_session_detection_pipeline(session)
        pipeline += session_detection_pipeline
        session_detection_nodes.append(output_node)

    pipeline += Pipeline(
        [
            # Load camera configuration.
            node(
                CameraConfig.load_partitioned,
                dict(
                    camera_xml="auto_camera_config",
                ),
                "camera_config",
            ),
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
            # Add plot numbers.
            node(
                FieldConfig.from_yml,
                "top_field_config_raw",
                "top_field_config",
            ),
            node(
                FieldConfig.from_yml,
                "middle_field_config_raw",
                "middle_field_config",
            ),
            node(
                FieldConfig.from_yml,
                "bottom_field_config_raw",
                "bottom_field_config",
            ),
            node(
                find_detections_in_plots_pre_september,
                dict(
                    detections="filtered_detection_results",
                    plot_boundaries="plot_borders_top",
                    field_config="top_field_config",
                ),
                "detection_results_plot_num_top_pre_sep",
            ),
            node(
                find_detections_in_plots_post_september,
                dict(
                    detections="filtered_detection_results",
                    plot_boundaries="plot_borders_top_sep",
                    field_config="top_field_config",
                ),
                "detection_results_plot_num_top_post_sep",
            ),
            node(
                find_detections_in_plots_pre_september,
                dict(
                    detections="filtered_detection_results",
                    plot_boundaries="plot_borders_middle",
                    field_config="middle_field_config",
                ),
                "detection_results_plot_num_middle_pre_sep",
            ),
            node(
                find_detections_in_plots_post_september,
                dict(
                    detections="filtered_detection_results",
                    plot_boundaries="plot_borders_middle_sep",
                    field_config="middle_field_config",
                ),
                "detection_results_plot_num_middle_post_sep",
            ),
            node(
                find_detections_in_plots_pre_september,
                dict(
                    detections="filtered_detection_results",
                    plot_boundaries="plot_borders_bottom",
                    field_config="bottom_field_config",
                ),
                "detection_results_plot_num_bottom_pre_sep",
            ),
            node(
                find_detections_in_plots_post_september,
                dict(
                    detections="filtered_detection_results",
                    plot_boundaries="plot_borders_bottom_sep",
                    field_config="bottom_field_config",
                ),
                "detection_results_plot_num_bottom_post_sep",
            ),
            node(
                lambda *d: pd.concat(d),
                [
                    "detection_results_plot_num_top_pre_sep",
                    "detection_results_plot_num_middle_pre_sep",
                    "detection_results_plot_num_bottom_pre_sep",
                    "detection_results_plot_num_top_post_sep",
                    "detection_results_plot_num_middle_post_sep",
                    "detection_results_plot_num_bottom_post_sep",
                ],
                "detection_results_plot_num",
            ),
            # Also extract the detections within the GT sampling regions.
            node(
                find_detections_in_gt_sampling_regions_pre_september,
                dict(
                    detections="filtered_detection_results",
                    gt_sampling_regions="plot_gt_sample_locations_top",
                    field_config="top_field_config",
                    ground_truth="gt_combined",
                ),
                "detection_results_gt_sample_top_pre_sep",
            ),
            node(
                find_detections_in_gt_sampling_regions_post_september,
                dict(
                    detections="filtered_detection_results",
                    gt_sampling_regions="plot_gt_sample_locations_top_sep",
                    field_config="top_field_config",
                    ground_truth="gt_combined",
                ),
                "detection_results_gt_sample_top_post_sep",
            ),
            node(
                find_detections_in_gt_sampling_regions_pre_september,
                dict(
                    detections="filtered_detection_results",
                    gt_sampling_regions="plot_gt_sample_locations_middle",
                    field_config="middle_field_config",
                    ground_truth="gt_combined",
                ),
                "detection_results_gt_sample_middle_pre_sep",
            ),
            node(
                find_detections_in_gt_sampling_regions_post_september,
                dict(
                    detections="filtered_detection_results",
                    gt_sampling_regions="plot_gt_sample_locations_middle_sep",
                    field_config="middle_field_config",
                    ground_truth="gt_combined",
                ),
                "detection_results_gt_sample_middle_post_sep",
            ),
            node(
                find_detections_in_gt_sampling_regions_pre_september,
                dict(
                    detections="filtered_detection_results",
                    gt_sampling_regions="plot_gt_sample_locations_bottom",
                    field_config="bottom_field_config",
                    ground_truth="gt_combined",
                ),
                "detection_results_gt_sample_bottom_pre_sep",
            ),
            node(
                find_detections_in_gt_sampling_regions_post_september,
                dict(
                    detections="filtered_detection_results",
                    gt_sampling_regions="plot_gt_sample_locations_bottom_sep",
                    field_config="bottom_field_config",
                    ground_truth="gt_combined",
                ),
                "detection_results_gt_sample_bottom_post_sep",
            ),
            node(
                lambda *d: pd.concat(d),
                [
                    "detection_results_gt_sample_top_pre_sep",
                    "detection_results_gt_sample_middle_pre_sep",
                    "detection_results_gt_sample_bottom_pre_sep",
                    "detection_results_gt_sample_top_post_sep",
                    "detection_results_gt_sample_middle_post_sep",
                    "detection_results_gt_sample_bottom_post_sep",
                ],
                "detection_results_gt_sample",
            ),
            # Convert detections to shapefile.
            node(
                flowers_to_shapefile,
                "detection_results_plot_num",
                "bounding_box_shapes",
            ),
            # Compute the flower counts.
            node(
                compute_counts,
                "detection_results_plot_num",
                "counting_results_no_dap_no_plot",
            ),
            node(
                add_plot_index,
                "counting_results_no_dap_no_plot",
                "counting_results_no_dap",
            ),
            node(
                add_dap_counting,
                dict(
                    counting_results="counting_results_no_dap",
                    field_planted_date="params:v2_field_planted_date",
                ),
                "counting_results",
            ),
            # Compute counts for the GT sample areas too.
            node(
                compute_counts,
                "detection_results_gt_sample",
                "counting_results_gt_sample_no_dap_no_plot",
            ),
            node(
                add_plot_index,
                "counting_results_gt_sample_no_dap_no_plot",
                "counting_results_gt_sample_no_dap",
            ),
            node(
                add_dap_counting,
                dict(
                    counting_results="counting_results_gt_sample_no_dap",
                    field_planted_date="params:v2_field_planted_date",
                ),
                "counting_results_gt_sample",
            ),
        ]
    )

    return pipeline + _create_analysis_pipeline()
