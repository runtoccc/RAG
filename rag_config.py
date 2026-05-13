from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "papers_dir": "data/papers",
        "chroma_dir": "data/vector_db/chroma",
        "kg_dir": "kg",
    },
    "chunking": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
    },
    "retrieval": {
        "top_k": 5,
    },
    "embedding": {
        "provider": "sentence_transformers",
        "model": "intfloat/multilingual-e5-small",
        "local_dir": "models/embedding/intfloat-multilingual-e5-small",
        "cache_dir": "models/embedding/cache",
        "normalize_embeddings": True,
        "batch_size": 64,
    },
    "llm": {
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "temperature": 0.2,
        "max_tokens": 2048,
    },
}


def load_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    path = Path(config_path)
    config = _deep_copy(DEFAULT_CONFIG)

    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        _deep_update(config, loaded)

    return config


def project_path(relative_or_absolute: str | Path) -> Path:
    path = Path(relative_or_absolute)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def _deep_copy(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _deep_copy(item) if isinstance(item, dict) else item
        for key, item in value.items()
    }


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base
