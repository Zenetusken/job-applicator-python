"""Structured logging setup."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure structured logging with rich output."""
    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
        )
    except ImportError:
        handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
    )

    logger = logging.getLogger("job_applicator")
    logger.setLevel(level.upper())
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance."""
    return logging.getLogger(f"job_applicator.{name}")
