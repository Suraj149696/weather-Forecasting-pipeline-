"""
src/utils/config.py
-------------------
Reads environment variables from config/.env
Usage:
    from src.utils.config import config
    DB_URL = config("DB_URL", default="sqlite:///local.db")
"""

import os
from pathlib import Path

# ── Auto-load config/.env on first import ────────────────────────────────────
_ENV_PATH = Path(__file__).resolve().parents[2] / "config" / ".env"

def _load_env(path: Path):
    """Minimal .env parser — no external dependency needed."""
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip blank lines and comments
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key   = key.strip()
            value = value.strip().strip('"').strip("'")
            # Only set if not already in environment
            os.environ.setdefault(key, value)

_load_env(_ENV_PATH)


def config(key: str, default: str = None) -> str:
    """
    Fetch a config value from environment / .env file.

    Parameters
    ----------
    key     : environment variable name
    default : fallback value if key is not found

    Returns
    -------
    str value, or default if missing
    """
    value = os.environ.get(key, default)
    if value is None:
        raise KeyError(
            f"[config] Required key '{key}' not found in environment or config/.env"
        )
    return value
