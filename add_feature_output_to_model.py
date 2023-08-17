"""
Converts a default model to one with an additional features output.
"""


import argparse
from pathlib import Path
from ultralytics import YOLO
from src.yolov8_train.detection_model_with_features import (
    DetectionModelWithFeatures,
)
import torch


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Converts a default model to one with an additional"
        " features output."
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to the model to convert.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to save the converted model.",
        default=Path("best_model_with_features.pt"),
    )
    return parser


def main() -> None:
    cli_args = _make_parser().parse_args()
    model_path = cli_args.model_path

    # Load the model.
    original_model = YOLO(model=model_path.as_posix())
    # Add the features output.
    with_features = DetectionModelWithFeatures.from_detection_model(
        original_model.model
    )

    torch.save({"model": with_features}, cli_args.output.as_posix())


if __name__ == "__main__":
    main()
