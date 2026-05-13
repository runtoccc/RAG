from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from get_embedding_function import get_embedding_function
from rag_config import load_config, project_path


REQUIRED_PATTERNS = [
    "config.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "1_Pooling/config.json",
]


def main():
    config = load_config()
    embedding_config = config["embedding"]
    provider = embedding_config.get("provider")
    if provider == "hashing":
        print("Current embedding provider is hashing. No model weights are required.")
        return

    if provider not in {"sentence_transformers", "local"}:
        print(f"Current embedding provider is {provider}. No local download is needed.")
        return

    repo_id = embedding_config["model"]
    local_dir = project_path(embedding_config["local_dir"])
    cache_dir = project_path(embedding_config.get("cache_dir", "models/embedding/cache"))

    print(f"Downloading embedding model: {repo_id}")
    print(f"Target local dir: {local_dir}")

    if local_dir.exists() and not is_complete_model_dir(local_dir):
        print("Removing incomplete local model directory...")
        shutil.rmtree(local_dir)

    local_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        cache_dir=str(cache_dir),
        allow_patterns=REQUIRED_PATTERNS,
        resume_download=True,
        local_dir_use_symlinks=False,
    )

    missing_files = get_missing_files(local_dir)
    if missing_files:
        raise RuntimeError(
            "Embedding model download is incomplete. Missing files: "
            + ", ".join(missing_files)
        )

    get_embedding_function(config)
    print(f"Embedding model ready: {repo_id}")


def is_complete_model_dir(local_dir: Path) -> bool:
    return not get_missing_files(local_dir)


def get_missing_files(local_dir: Path) -> list[str]:
    return [
        pattern
        for pattern in REQUIRED_PATTERNS
        if not (local_dir / pattern).exists()
    ]


if __name__ == "__main__":
    main()
