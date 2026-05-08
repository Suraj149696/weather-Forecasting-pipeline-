"""
src/utils/logger.py
===================
Shared logger factory used across the pipeline.
Usage:
    from src.utils.logger import get_logger
    logger = get_logger("train", LOG_PATH)
"""

import logging
from pathlib import Path


def get_logger(name: str, log_path: Path = None) -> logging.Logger:
    """
    Returns a logger that writes to both the console and an optional log file.

    Parameters
    ----------
    name     : logger name (e.g. "train", "extract", "load")
    log_path : Path to the .log file.  Pass None for console-only.
    """
    logger = logging.getLogger(name)

    if logger.handlers:          # avoid adding duplicate handlers on re-import
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(name)-10s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (optional)
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
