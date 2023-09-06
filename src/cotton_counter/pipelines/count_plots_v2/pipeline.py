"""
Version two of the pipeline for auto-counting.
"""
from functools import partial, reduce
from operator import add
from typing import Tuple

import pandas as pd
from kedro.pipeline import Pipeline, node

from ..common import (
    add_dap_counting,
    add_dap_ground_truth,
    collect_session_results,
    compute_counts,
    filter_low_confidence,
    merge_ground_truth,
    plot_ground_truth_regression,
)
from .field_config import FieldConfig
from .nodes import (
    add_plot_index,
    detect_flowers,
    find_detections_in_gt_sampling_regions,
    find_detections_in_plots,
    find_image_extents,
    flowers_to_geographic,
    flowers_to_shapefile,
    load_ground_truth,
    plot_ground_truth_vs_predicted,
    prune_duplicate_detections,
)
from .camera_utils import CameraConfig

SESSIONS = {
    # "2023-07-27",
    # "2023-08-01",
    # "2023-08-03",
    # "2023-08-07",
    # "2023-08-10",
    # "2023-08-14",
    # "2023-08-18",
    "2023-08-21",
    # "2023-08-24",
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
                # Detect flowers.
                node(
                    detect_flowers_session,
                    dict(
                        images=f"images_{session}",
                        weights_file="params:model_weights_file",
                        batch_size="params:prediction_batch_size",
                    ),
                    f"detections_px_{session}",
                ),
                # Load associated camera configuration.
                node(
                    CameraConfig.load,
                    dict(
                        camera_xml=f"{session}_auto_camera_config",
                    ),
                    f"camera_config_{session}",
                ),
                # Convert to geographic coordinates.
                node(
                    flowers_to_geographic,
                    dict(
                        detections=f"detections_px_{session}",
                        camera_config=f"camera_config_{session}",
                        dem_dataset=f"dem_{session}",
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
    for session in SESSIONS:
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
                    camera_config=f"camera_config_{session}",
                    dem_dataset=f"dem_{session}",
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
            # Convert to shapefile.
            node(
                flowers_to_shapefile,
                "filtered_detection_results",
                "bounding_box_shapes",
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
                find_detections_in_plots,
                dict(
                    detections="filtered_detection_results",
                    plot_boundaries="plot_borders_top",
                    field_config="top_field_config",
                ),
                "detection_results_plot_num_top",
            ),
            node(
                find_detections_in_plots,
                dict(
                    detections="detection_results_plot_num_top",
                    plot_boundaries="plot_borders_middle",
                    field_config="middle_field_config",
                ),
                "detection_results_plot_num_top_middle",
            ),
            node(
                find_detections_in_plots,
                dict(
                    detections="detection_results_plot_num_top_middle",
                    plot_boundaries="plot_borders_bottom",
                    field_config="bottom_field_config",
                ),
                "detection_results_plot_num",
            ),
            # Also extract the detections within the GT sampling regions.
            node(
                find_detections_in_gt_sampling_regions,
                dict(
                    detections="filtered_detection_results",
                    gt_sampling_regions="plot_gt_sample_locations_top",
                    field_config="top_field_config",
                    ground_truth="gt_combined",
                ),
                "detection_results_gt_sample_top",
            ),
            node(
                find_detections_in_gt_sampling_regions,
                dict(
                    detections="detection_results_gt_sample_top",
                    gt_sampling_regions="plot_gt_sample_locations_middle",
                    field_config="middle_field_config",
                    ground_truth="gt_combined",
                ),
                "detection_results_gt_sample_top_middle",
            ),
            node(
                find_detections_in_gt_sampling_regions,
                dict(
                    detections="detection_results_gt_sample_top_middle",
                    gt_sampling_regions="plot_gt_sample_locations_bottom",
                    field_config="bottom_field_config",
                    ground_truth="gt_combined",
                ),
                "detection_results_gt_sample",
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
            # Plot the ground truth regression.
            node(
                merge_ground_truth,
                dict(
                    counting_results="counting_results_gt_sample",
                    ground_truth="gt_combined",
                ),
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
        ]
    )

    return pipeline
