from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from populate_database import add_to_chroma, clear_database, load_documents
from rag_config import load_config


def main() -> None:
    args = parse_args()
    if not args.rebuild_chroma:
        print("[chroma] Skipping Chroma rebuild. Pass --rebuild-chroma to rebuild.")
        return
    config = load_config()
    clear_database(config)
    chunks = load_documents(config)
    add_to_chroma(chunks, config, batch_size=args.batch_size)
    print(f"[chroma] rebuilt chunks={len(chunks)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chroma from validated OpenScholar-style passages.")
    parser.add_argument("--rebuild-chroma", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    main()
