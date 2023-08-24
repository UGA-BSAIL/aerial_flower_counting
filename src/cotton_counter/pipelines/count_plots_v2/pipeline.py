"""
Version two of the pipeline for auto-counting.
"""

from kedro.pipeline import Pipeline, node
from typing import Tuple
from functools import partial
from .nodes import (
    detect_flowers,
    flowers_to_geographic,
    CameraConfig,
    flowers_to_shapefile,
)
from ..common import collect_session_results, filter_low_confidence


SESSIONS = {"2023-08-14"}
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
                        parameters="user_camera_config",
                    ),
                    f"camera_config_{session}",
                ),
                # Convert to geographic coordinates.
                node(
                    flowers_to_geographic,
                    dict(
                        detections=f"detections_px_{session}",
                        camera_config=f"camera_config_{session}",
                    ),
                    output_node,
                ),
            ]
        ),
        output_node,
    )


def create_pipeline(**kwargs) -> Pipeline:
    pipeline = Pipeline([])

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
        ]
    )

    return pipeline
