"""Single source of truth for configuration. Nothing hard-codes a value that
belongs in config/config.yaml."""
from __future__ import annotations

import os
import functools
import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(_REPO_ROOT, "config", "config.yaml")


def repo_root() -> str:
    return _REPO_ROOT


@functools.lru_cache(maxsize=None)
def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)
