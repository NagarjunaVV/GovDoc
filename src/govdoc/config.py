"""Configuration loader and utilities for govdoc-rag."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def get_repo_root() -> Path:
    """Return the absolute path to the repository root."""
    return _REPO_ROOT


def load_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """Load configuration dictionary from YAML file."""
    if config_path is None:
        config_path = _REPO_ROOT / "config" / "config.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = _REPO_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_config_hash(cfg: Dict[str, Any] | None = None) -> str:
    """Compute deterministic SHA-256 hash of configuration."""
    if cfg is None:
        cfg = load_config()
    serialized = json.dumps(cfg, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def resolve_path(rel_or_abs_path: str | Path) -> Path:
    """Resolve a path relative to the repo root if it is not absolute."""
    p = Path(rel_or_abs_path)
    if p.is_absolute():
        return p
    return _REPO_ROOT / p
