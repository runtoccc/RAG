from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUTOSKG_ROOT = PROJECT_ROOT / "data" / "autoskg" / "kg_project"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_project_config() -> dict[str, Any]:
    load_dotenv()
    config = default_config()
    if CONFIG_PATH.exists():
        loaded = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        deep_update(config, loaded)
    return config


def autoskg_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_project_config()
    llm_config = config.get("llm", {})
    section = config.get("autoskg", {})
    return {
        "root_dir": section.get("root_dir", "data/autoskg/kg_project"),
        "input_dir": section.get("input_dir", config.get("paths", {}).get("papers_dir", "data/papers")),
        "tei_dir": section.get("tei_dir", "data/interim/tei"),
        "grobid_url": section.get("grobid_url", "http://localhost:8070/api/processFulltextDocument"),
        "prefer_existing_tei": bool(section.get("prefer_existing_tei", True)),
        "license_policy": section.get("license_policy", "allow_all"),
        "unpaywall_email": section.get("unpaywall_email", "ai@plantscience.ai"),
        "llm_provider": llm_config.get("provider", "deepseek"),
        "llm_model": section.get("llm_model", llm_config.get("model", "deepseek-v4-flash")),
        "llm_base_url": section.get("llm_base_url", llm_config.get("base_url", "https://api.deepseek.com")),
        "llm_api_key_env": section.get("llm_api_key_env", llm_config.get("api_key_env", "DEEPSEEK_API_KEY")),
        "llm_temperature": float(section.get("llm_temperature", llm_config.get("temperature", 0.2))),
        "llm_max_tokens": int(section.get("llm_max_tokens", llm_config.get("max_tokens", 2048))),
    }


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_deepseek_env(settings: dict[str, Any]) -> None:
    api_key_env = settings["llm_api_key_env"]
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing DeepSeek API key. Set {api_key_env} in .env or the shell.")

    os.environ.setdefault("DEEPSEEK_API_KEY", api_key)
    os.environ.setdefault("DEEPSEEK_BASE_URL", settings["llm_base_url"])
    os.environ.setdefault("DEEPSEEK_MODEL", settings["llm_model"])


def load_dotenv(env_path: str | Path | None = None) -> None:
    path = Path(env_path) if env_path else PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def default_config() -> dict[str, Any]:
    return {
        "paths": {
            "papers_dir": "data/papers",
        },
        "llm": {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com",
            "api_key_env": "DEEPSEEK_API_KEY",
            "temperature": 0.2,
            "max_tokens": 2048,
        },
        "autoskg": {
            "root_dir": "data/autoskg/kg_project",
            "input_dir": "data/papers",
            "tei_dir": "data/autoskg/tei",
            "grobid_url": "http://localhost:8070/api/processFulltextDocument",
            "prefer_existing_tei": True,
            "license_policy": "allow_all",
            "unpaywall_email": "ai@plantscience.ai",
            "llm_model": "deepseek-v4-flash",
            "llm_base_url": "https://api.deepseek.com",
            "llm_api_key_env": "DEEPSEEK_API_KEY",
            "llm_temperature": 0.2,
            "llm_max_tokens": 2048,
        },
    }


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base
