"""
utils/yaml_loader.py

Low-level YAML file loader for the Multi-Agent Procurement System.

Responsibilities:
    - Read a YAML file from disk and return it as a Python dict
    - Raise a clear custom exception if the file is missing or malformed
    - Stay generic — no knowledge of project structure or path constants
      (project_paths.py and settings.py handle those concerns)

This module is intentionally thin.
Caching is NOT applied here — it belongs in settings.py (lru_cache getters).
One function, one responsibility.
"""

import logging

import yaml
from pathlib import Path

from procurement_system.exceptions import (
    ConfigFileNotFoundError,
    InvalidConfigError,
)

logger = logging.getLogger(__name__)


def load_yaml(path: str | Path) -> dict:
    """
    Read a YAML file and return its contents as a Python dict.

    Args:
        path: Absolute or relative path to the YAML file.
              Pass str(Path) from project_paths.py for safety.

    Returns:
        Parsed YAML content as a dict.

    Raises:
        ConfigFileNotFoundError: File does not exist at the given path.
        InvalidConfigError:      File exists but is empty, malformed,
                                 or does not parse to a dict.

    Example:
        from procurement_system.utils.yaml_loader import load_yaml
        from procurement_system.project_paths import ENTERPRISE_BUYING_RULES

        data = load_yaml(ENTERPRISE_BUYING_RULES)  # Path or str
    """
    logger.debug(f"Loading YAML file: {path}")

    # ── File existence ────────────────────────────────────
    try:
        with open(path, encoding="utf-8") as f:  # Path and str both accepted
            raw = f.read()
    except FileNotFoundError:
        raise ConfigFileNotFoundError(path=path)
    except OSError as e:
        raise InvalidConfigError(
            file=path,
            reason=f"Could not read file: {e}"
        ) from e

    # ── Empty file ────────────────────────────────────────
    if not raw.strip():
        raise InvalidConfigError(
            file=path,
            reason="File is empty"
        )

    # ── YAML parsing ──────────────────────────────────────
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise InvalidConfigError(
            file=path,
            reason=f"YAML parse error: {e}"
        ) from e

    # ── Type check — must be a dict ───────────────────────
    if not isinstance(data, dict):
        raise InvalidConfigError(
            file=path,
            reason=(
                f"Expected a YAML mapping (dict) at root level, "
                f"got {type(data).__name__}"
            )
        )

    logger.debug(f"YAML loaded successfully: {path} ({len(data)} top-level keys)")
    return data
