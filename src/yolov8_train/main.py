from .yolov8_wandb import register_callbacks
from ultralytics import YOLO
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional


def _make_parser() -> ArgumentParser:
    """
    Returns:
        The parser for CLI arguments.

    """
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        required=True,
        type=Path,
        help="Path to the data file.",
    )
    parser.add_argument("-n", "--name", type=str, help="Name of the run")
    parser.add_argument(
        "-m",
        "--model",
        default="yolov8s.pt",
        help="Path to the model to use.",
    )

    return parser


def train(dataset: Path, *, model: str, name: Optional[str] = None) -> None:
    """
    Trains the model.

    Args:
        dataset: Path to the dataset descriptor file.
        model: The model to use.
        name: The name of this run.

    """
    # Load a model.
    model = YOLO(model)
    register_callbacks(model)
    # Train the model.
    model.train(
        data=dataset,
        epochs=100,
        imgsz=720,
        project="flower_counting",
        name=name,
        batch=-1,
    )


def main() -> None:
    """
    Main function.

    """
    parser = _make_parser()
    args = parser.parse_args()
    train(args.data, model=args.model, name=args.name)
