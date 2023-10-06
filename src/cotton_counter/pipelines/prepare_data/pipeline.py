"""
Pipeline for performing one-time data preparation.
"""

from kedro.pipeline import Pipeline, node
from ..common import SESSIONS, merge_dicts
from .nodes import compute_excess_green
from typing import Dict
from functools import partial


def _merge_image_datasets() -> Pipeline:
    def _prefix_keys(session_: str, partitions: Dict) -> Dict:
        # Add session names as prefixes to partitioned keys.
        return {f"{session_}_{k}": v for k, v in partitions.items()}

    # Nodes that add session names to the image keys.
    session_prefix_nodes = []
    session_prefix_node_names = []
    for session in SESSIONS:
        output_node = f"prefixed_images_{session}"
        session_prefix_node_names.append(output_node)
        session_prefix_nodes.append(
            node(
                partial(_prefix_keys, session),
                f"images_{session}",
                output_node,
            )
        )

    return Pipeline(
        session_prefix_nodes
        + [node(merge_dicts, session_prefix_node_names, "combined_images")]
    )


def create_pipeline(**kwargs) -> Pipeline:
    pipeline = _merge_image_datasets()
    return pipeline + Pipeline(
        [
            # Extract excess green.
            node(compute_excess_green, "combined_images", "excess_green")
        ]
    )
