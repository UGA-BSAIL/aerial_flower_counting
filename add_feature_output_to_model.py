"""
Converts a default model to one with an additional features output.
"""


import argparse
from pathlib import Path
import shutil

from ultralytics import YOLO
from loguru import logger
import numpy as np


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


def _export(model: YOLO, *, output_dir: Path) -> None:
    """
    Exports the models as TF SavedModels.

    Args:
        model: The model to export.
        output_dir: The directory to export the models to.

    """
    output_dir.mkdir(exist_ok=True)

    logger.debug("Exporting training model...")
    export_dir = model.export(format="saved_model", imgsz=(540, 960), batch=32)
    shutil.copytree(export_dir, output_dir / "training")

    logger.debug("Exporting primary inference model...")
    export_dir = model.export(format="saved_model", imgsz=(540, 960), batch=1)
    shutil.copytree(export_dir, output_dir / "inference")

    logger.debug("Exporting small inference model...")
    export_dir = model.export(format="saved_model", imgsz=(256, 256), batch=1)
    shutil.copytree(export_dir, output_dir / "inference_small")


def main() -> None:
    cli_args = _make_parser().parse_args()
    model_path = cli_args.model_path

    # Our custom Ultralytics version already contains functionality
    # for a separate embeddings output, so adding it is as simple
    # as loading the weights.
    model = YOLO(model=model_path.as_posix())
    model.model.embedding_layer = 5
    model.model.output_raw_embeddings = True

    # Test it on some data to make sure it outputs the right thing.
    logger.info("Testing updated model...")
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    outputs = model.predict(test_image)[0]
    assert len(outputs) == 2
    logger.info("Model update successful!")

    _export(model, output_dir=cli_args.output)


if __name__ == "__main__":
    main()
