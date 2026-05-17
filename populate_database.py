from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain.vectorstores.chroma import Chroma

from get_embedding_function import get_embedding_function
from rag_config import load_config, project_path


DEFAULT_PASSAGES_PATH = "data/passages/scientific_passages.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Chroma add_documents batch size.")
    args = parser.parse_args()
    config = load_config()

    if args.reset:
        print("Clearing Chroma database")
        clear_database(config)

    chunks = load_documents(config)
    add_to_chroma(chunks, config, batch_size=args.batch_size)


def load_documents(config: dict[str, Any] | None = None) -> list[Document]:
    config = config or load_config()
    passages_path = project_path(
        config["paths"].get("passages_jsonl", DEFAULT_PASSAGES_PATH)
    )
    if not passages_path.exists() or passages_path.stat().st_size == 0:
        raise RuntimeError(
            f"Scientific passages file does not exist or is empty: {passages_path}. "
            "Run scripts/01_parse_pdf_to_tei.py through scripts/04_build_scientific_passages.py first."
        )

    # TODO: For 100K-paper scale, replace this list loader with a streaming iterator.
    documents: list[Document] = []
    with passages_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            documents.append(scientific_passage_record_to_document(record, line_number))

    print(f"Loaded scientific passages: {len(documents)} from {passages_path}")
    return documents


def scientific_passage_record_to_document(record: dict[str, Any], line_number: int = 0) -> Document:
    passage_id = normalize_metadata_text(record.get("passage_id")) or f"passage:{line_number}"
    paper_id = normalize_metadata_text(record.get("paper_id")) or "unknown_paper"
    title = normalize_metadata_text(record.get("title")) or "Unknown title"
    text = normalize_passage_text(record.get("text") or "")
    page_number = normalize_page_number(record.get("page"))
    source_file = normalize_metadata_text(record.get("source_file")) or f"{paper_id}.pdf"
    doi = normalize_metadata_text(record.get("doi"))
    embedding_text = str(record.get("embedding_text") or f"{title}\n\n{text}")
    if record.get("index_ready") is not True:
        raise RuntimeError(f"Refusing to index non-index-ready passage: {passage_id}")
    if embedding_text != f"{title}\n\n{text}":
        raise RuntimeError(f"Refusing to index passage with invalid embedding_text: {passage_id}")

    metadata = {
        "chunk_id": passage_id,
        "id": passage_id,
        "passage_id": passage_id,
        "paper_id": paper_id,
        "title": title,
        "section": normalize_metadata_text(record.get("section_title")),
        "section_type": normalize_metadata_text(record.get("section_type")),
        "block_index": record.get("block_index"),
        "block_words": record.get("block_words"),
        "chunk_style": normalize_metadata_text(record.get("chunk_style")),
        "index_ready": record.get("index_ready"),
        "source_file": source_file,
        "page_number": page_number,
        "doi": doi,
        "is_reference_section": False,
        "chunk_source": "scientific_passages_jsonl",
    }
    return Document(page_content=embedding_text, metadata=metadata)


def normalize_passage_text(text: str) -> str:
    text = str(text).replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_metadata_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_page_number(page: object) -> int | str:
    if page in (None, ""):
        return "unknown"
    try:
        return int(page)
    except (TypeError, ValueError):
        return str(page)


def add_to_chroma(
    chunks: list[Document],
    config: dict[str, Any] | None = None,
    batch_size: int = 1000,
) -> None:
    config = config or load_config()
    chroma_path = project_path(config["paths"]["chroma_dir"])
    chroma_path.mkdir(parents=True, exist_ok=True)

    db = Chroma(
        persist_directory=str(chroma_path),
        embedding_function=get_embedding_function(config),
    )

    # NOTE: db.get(include=[]) is simple and fine for this demo, but can become slow
    # on large Chroma collections. At 100K-paper scale, prefer versioned collections
    # or a batch existence-check strategy instead of loading all ids at once.
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [
        chunk for chunk in chunks if chunk.metadata["chunk_id"] not in existing_ids
    ]

    if new_chunks:
        print(f"Adding new chunks: {len(new_chunks)}")
        for start in range(0, len(new_chunks), batch_size):
            batch = new_chunks[start : start + batch_size]
            print(f"Adding batch {start // batch_size + 1}: {len(batch)} chunks")
            db.add_documents(
                batch,
                ids=[chunk.metadata["chunk_id"] for chunk in batch],
            )
        if hasattr(db, "persist"):
            db.persist()
    else:
        print("No new chunks to add")


def clear_database(config: dict[str, Any] | None = None) -> None:
    config = config or load_config()
    chroma_path = project_path(config["paths"]["chroma_dir"])
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)


if __name__ == "__main__":
    main()
