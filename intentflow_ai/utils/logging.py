"""Logging helpers configured with structured context."""

from __future__ import annotations

import logging
from typing import Optional

from rich.logging import RichHandler


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-level logger with Rich formatting."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(rich_tracebacks=True)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
