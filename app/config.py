import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "database": "data/faces.db",
    "thumbnails_dir": "data/thumbs",
    "models_dir": "data/models",
    "threshold": 0.40,
    "max_side": 1600,
    "det_size": 640,
    "download_workers": 6,
    "providers": ["CPUExecutionProvider"],
    "sources": [],
}


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(path: str | None = None) -> dict:
    path = path or os.environ.get("FACESCAN_CONFIG", "config.yaml")
    cfg_path = resolve(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"No existe {cfg_path}. Copia config.example.yaml a config.yaml.")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = {**DEFAULTS, **(yaml.safe_load(f) or {})}
    return cfg
