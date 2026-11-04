"""Read project configuration without accessing data sources."""
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and reject an empty or invalid root."""
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError("Configuration must be a YAML mapping")
    return config

