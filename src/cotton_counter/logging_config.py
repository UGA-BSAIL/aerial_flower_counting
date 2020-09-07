"""
Utilities for configuring logging.
"""


import logging

from loguru import logger


class PropagateHandler(logging.Handler):
    """
    Handler that propagates `Loguru` messages to a vanilla logging handler.
    This is so `Loguru` can coexist nicely with `Kedro`'s built-in logging.
    """

    def emit(self, record):
        logging.getLogger(record.name).handle(record)


def init_logging() -> None:
    """
    Initializes logging for the package.
    """
    logger.add(PropagateHandler(), format="{message}")
