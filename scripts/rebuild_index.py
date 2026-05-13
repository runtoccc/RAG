from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from populate_database import add_to_chroma, clear_database, load_documents, split_documents
from rag_config import load_config


def main():
    config = load_config()
    clear_database(config)
    documents = load_documents(config)
    chunks = split_documents(documents, config)
    add_to_chroma(chunks, config)


if __name__ == "__main__":
    main()
