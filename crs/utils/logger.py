"""
src/utils/logger.py
-------------------
Simple logger that writes to both console and logs/pipeline.log
"""

import logging
import os

LOG_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")


def get_logger(name: str = "weather_pipeline") -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ──────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ── File handler (logs/pipeline.log) ─────────────────────────────────────
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
