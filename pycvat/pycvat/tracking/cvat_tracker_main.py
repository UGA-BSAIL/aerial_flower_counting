"""
Main file for CVAT tracker utility.
"""


from argparse import ArgumentParser

from loguru import logger

from ..dataset.api import make_api_client
from ..dataset.task import Task
from .tracker import Tracker


def _make_parser() -> ArgumentParser:
    """
    Returns:
        The parser to use for parsing CLI arguments.
    """
    parser = ArgumentParser(
        prog="pycvat",
        description="Program that uses tracking to propagate annotations from "
        "one frame to the next.",
    )

    parser.add_argument(
        "-u",
        "--username",
        required=True,
        help="The username to use for logging into CVAT.",
    )
    parser.add_argument(
        "-p",
        "--password",
        required=True,
        help="The password to use for logging into CVAT.",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8080/api/v1/",
        help="The CVAT API endpoint to connect to.",
    )

    parser.add_argument(
        "-t",
        "--task-id",
        type=int,
        help="The numerical ID of the task to modify.",
    )
    parser.add_argument(
        "-j",
        "--job-num",
        type=int,
        default=0,
        help="The job number within the task to modify.",
    )
    parser.add_argument(
        "-f", "--frame", type=int, help="The frame to start tracking at."
    )

    parser.add_argument(
        "--show-result",
        action="store_true",
        help="Enables a debugging display that shows the transformed "
        "annotations.",
    )

    return parser


@logger.catch()
def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    api_client = make_api_client(
        username=args.username, password=args.password, host=args.url
    )
    with Task.init_and_upload(
        task_id=args.task_id, api_client=api_client
    ) as task:
        # Update the annotations with tracking.
        tracker = Tracker(task, job_num=args.job_num)
        tracker.track_forward(
            start_frame=args.frame, show_result=args.show_result
        )
