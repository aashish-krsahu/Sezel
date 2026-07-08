"""Configuration loading from YAML."""

import yaml
from pathlib import Path
from typing import Any


class Config:
    """Configuration manager."""

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def load_models(config_dir: str | Path = "config") -> dict[str, Any]:
        """Load models.yaml from the config directory."""
        config_dir = Path(config_dir)
        models_path = config_dir / "models.yaml"
        return Config.load(models_path)

