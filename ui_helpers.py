from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import chromadb

from env_loader import load_dotenv
from rag_config import BASE_DIR, load_config, project_path


DEFAULT_COLLECTION_NAME = "langchain"


def get_config() -> dict[str, Any]:
    return load_config()


def get_papers(config: dict[str, Any] | None = None) -> list[Path]:
    config = config or load_config()
    papers_dir = project_path(config["paths"]["papers_dir"])
    if not papers_dir.exists():
        return []
    return sorted(papers_dir.glob("*.pdf"), key=lambda path: path.name.lower())


def get_vector_db_status(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_config()
    chroma_path = project_path(config["paths"]["chroma_dir"])
    status: dict[str, Any] = {
        "path": chroma_path,
        "exists": chroma_path.exists(),
        "chunk_count": 0,
        "collection_name": DEFAULT_COLLECTION_NAME,
        "metadata_examples": [],
        "error": None,
    }

    if not chroma_path.exists():
        return status

    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection(DEFAULT_COLLECTION_NAME)
        status["chunk_count"] = collection.count()
        peek = collection.peek(limit=3)
        metadatas = peek.get("metadatas") or []
        documents = peek.get("documents") or []
        examples = []
        for index, metadata in enumerate(metadatas):
            examples.append(
                {
                    "metadata": metadata or {},
                    "snippet": make_snippet(documents[index] if index < len(documents) else ""),
                }
            )
        status["metadata_examples"] = examples
    except Exception as error:
        status["error"] = str(error)

    return status


def get_project_status(config: dict[str, Any] | None = None) -> dict[str, Any]:
    load_dotenv()
    config = config or load_config()
    papers = get_papers(config)
    vector_status = get_vector_db_status(config)

    return {
        "config": config,
        "papers": papers,
        "pdf_count": len(papers),
        "papers_dir": project_path(config["paths"]["papers_dir"]),
        "vector_db": vector_status,
        "deepseek_key_present": bool(os.getenv(config["llm"].get("api_key_env", "DEEPSEEK_API_KEY"))),
    }


def run_project_script(script_relative_path: str, timeout: int = 600) -> dict[str, Any]:
    script_path = BASE_DIR / script_relative_path
    if not script_path.exists():
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": f"Script not found: {script_path}",
        }

    process = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


def format_command_output(result: dict[str, Any]) -> str:
    parts = []
    if result.get("stdout"):
        parts.append(result["stdout"].strip())
    if result.get("stderr"):
        parts.append(result["stderr"].strip())
    if not parts:
        parts.append(f"Process exited with code {result.get('returncode')}")
    return "\n\n".join(parts)


def make_snippet(text: str, max_chars: int = 500) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."
