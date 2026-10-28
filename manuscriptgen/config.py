"""YAML configuration with defaults + dotted-key CLI overrides."""
import copy
from pathlib import Path

import yaml


class Cfg(dict):
    """dict with attribute access (recursively)."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v

    def __setattr__(self, k, v):
        self[k] = v


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(s):
    try:
        return yaml.safe_load(s)
    except Exception:
        return s


def load_config(path="config.yaml", overrides=None):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for item in overrides or []:
        key, _, val = item.partition("=")
        node = data
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(val)
    cfg = Cfg(data)
    cfg["_root"] = str(path.resolve().parent)
    return cfg


def resolve(cfg, p):
    """Resolve a path relative to the directory containing config.yaml."""
    p = Path(p)
    return p if p.is_absolute() else Path(cfg["_root"]) / p
