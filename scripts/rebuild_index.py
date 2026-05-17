from __future__ import annotations

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from populate_database import add_to_chroma, clear_database, load_documents
from rag_config import load_config


def main():
    parser = argparse.ArgumentParser(description="Rebuild the local Chroma index from scientific passages.")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()
    config = load_config()
    clear_database(config)
    chunks = load_documents(config)
    add_to_chroma(chunks, config, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
