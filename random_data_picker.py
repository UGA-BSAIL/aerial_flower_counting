"""
Tiny utility for choosing random field images.
"""

import random
from argparse import ArgumentParser
from pathlib import Path


def _choose_session(base_path: Path) -> Path:
    """
    Chooses a session randomly.

    Args:
        base_path: The base path where the session directories are.

    Returns:
        The session path.

    """
    sessions = [i for i in base_path.iterdir() if i.is_dir()]
    return random.choice(sessions)


def _choose_file(*, session_path: Path, image_path: Path) -> Path:
    """
    Chooses a file from a particular session.

    Args:
        session_path: The path of the session directory.
        image_path: The relative path from the session directory to the
            directory containing images.

    Returns:
        The path to the chosen image.

    """
    image_dir_path = session_path / image_path
    images = list(image_dir_path.iterdir())
    return random.choice(images)


def _make_parser() -> ArgumentParser:
    """
    Creates the parser to use for CLI arguments.

    Returns:
        The parser that it created.

    """
    parser = ArgumentParser(description="Chooses a field image randomly.")

    parser.add_argument(
        "base_path",
        type=Path,
        help="The path to the directory containing sessions.",
    )
    parser.add_argument(
        "-i",
        "--image-path",
        type=Path,
        default=Path("m100/raw"),
        help="Relative path from session directories to images.",
    )

    return parser


def main() -> None:
    parser = _make_parser()
    cli_args = parser.parse_args()

    session_dir = _choose_session(cli_args.base_path)
    image_file = _choose_file(
        session_path=session_dir, image_path=cli_args.image_path
    )
    print(image_file.relative_to(cli_args.base_path))


if __name__ == "__main__":
    main()
