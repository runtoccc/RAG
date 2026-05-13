from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from populate_database import add_to_chroma, load_documents, split_documents
from rag_config import load_config, project_path


def main():
    parser = argparse.ArgumentParser(
        description="Add newly collected weekly PDF papers to the RAG index."
    )
    parser.add_argument(
        "--incoming-dir",
        type=str,
        default=None,
        help="Optional folder containing new PDFs to copy into data/papers before indexing.",
    )
    args = parser.parse_args()

    config = load_config()
    if args.incoming_dir:
        copy_new_pdfs(Path(args.incoming_dir), project_path(config["paths"]["papers_dir"]))

    documents = load_documents(config)
    chunks = split_documents(documents, config)
    add_to_chroma(chunks, config)


def copy_new_pdfs(incoming_dir: Path, papers_dir: Path) -> list[Path]:
    papers_dir.mkdir(parents=True, exist_ok=True)
    copied_files = []

    for pdf_path in incoming_dir.glob("*.pdf"):
        target = papers_dir / pdf_path.name
        if not target.exists():
            shutil.copy2(pdf_path, target)
            copied_files.append(target)

    print(f"Copied {len(copied_files)} new PDFs into {papers_dir}")
    return copied_files


if __name__ == "__main__":
    main()
