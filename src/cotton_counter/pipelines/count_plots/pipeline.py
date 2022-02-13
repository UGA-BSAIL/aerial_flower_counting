"""
Pipeline for auto-counting.
"""


from functools import partial
from typing import Tuple

from kedro.pipeline import Pipeline, node

from .nodes import (
    FieldConfig,
    collect_session_results,
    create_per_plot_table,
    detect_flowers,
    filter_low_confidence,
)

SESSIONS = {
    "2021-08-30",
    "2021-09-03",
    "2021-09-06",
    "2021-09-10",
    "2021-09-13",
    "2021-09-17",
    "2021-09-24",
    "2021-09-27",
    "2021-10-01",
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
            # Create the output count table.
            node(
                FieldConfig,
                dict(
                    num_rows="params:field_num_rows",
                    first_row_num="params:field_first_row_num",
                    first_plot_num="params:field_first_plot_num",
                    empty_rows="params:field_empty_rows",
                ),
                "field_config",
                name="field_config",
            ),
            node(
                create_per_plot_table,
                dict(
                    counting_results="filtered_detection_results",
                    field_config="field_config",
                ),
                "plot_counts",
            ),
        ]
    )

    return pipeline
