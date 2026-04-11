"""Reusable YAML config loader for all PolySignalLab components."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)


def load_config(config_path: Path | str, required_keys: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config file and return it as a plain dict.

    Args:
        config_path: Path to the .yml file.
        required_keys: Top-level keys that must be present. Exits with an error
                       if any are missing.

    Returns:
        Parsed config dict.
    """
    path = Path(config_path)
    if not path.exists():
        sys.stderr.write(f"Config file not found: {path.resolve()}\n")
        sys.exit(1)

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        sys.stderr.write(f"Config parse error in {path}: {exc}\n")
        sys.exit(1)

    if not isinstance(raw, dict):
        sys.stderr.write(f"Config file must be a YAML mapping, got {type(raw).__name__}: {path}\n")
        sys.exit(1)

    if required_keys:
        missing = [k for k in required_keys if k not in raw]
        if missing:
            sys.stderr.write(f"Missing required config keys in {path}: {', '.join(missing)}\n")
            sys.exit(1)

    return raw


def get_nested(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dict keys, returning default if any key is missing."""
    current: Any = cfg
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
