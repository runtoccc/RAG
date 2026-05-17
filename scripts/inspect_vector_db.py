from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui_helpers import get_vector_db_status


def main():
    configure_stdout()
    status = get_vector_db_status()
    chunk_examples = []
    for example in status["metadata_examples"]:
        metadata = example.get("metadata") or {}
        chunk_examples.append(
            {
                "title": metadata.get("title"),
                "section": metadata.get("section"),
                "page_number": metadata.get("page_number"),
                "chunk_id": metadata.get("chunk_id") or metadata.get("id"),
                "is_reference_section": metadata.get("is_reference_section", False),
                "source_file": metadata.get("source_file"),
                "page_content": example.get("page_content_preview") or example.get("snippet", "")[:300],
            }
        )
    serializable = {
        "path": str(status["path"]),
        "exists": status["exists"],
        "collection_name": status["collection_name"],
        "chunk_count": status["chunk_count"],
        "chunk_examples": chunk_examples,
        "metadata_examples": status["metadata_examples"],
        "error": status["error"],
    }
    print(json.dumps(serializable, ensure_ascii=False, indent=2))


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    main()
