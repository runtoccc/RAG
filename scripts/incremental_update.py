from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from populate_database import add_to_chroma, load_documents
from rag_config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Add scientific passages to the RAG index."
    )
    parser.add_argument(
        "--incoming-dir",
        type=str,
        default=None,
        help="Deprecated. Run the PDF-to-passage pipeline before indexing instead.",
    )
    args = parser.parse_args()

    config = load_config()
    if args.incoming_dir:
        raise RuntimeError(
            "--incoming-dir is no longer supported. Add PDFs to data/papers, "
            "run scripts/01_parse_pdf_to_tei.py through scripts/04_build_scientific_passages.py, "
            "then run this script."
        )

    chunks = load_documents(config)
    add_to_chroma(chunks, config)


if __name__ == "__main__":
    main()
