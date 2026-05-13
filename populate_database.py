import argparse
import os
import re
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain.vectorstores.chroma import Chroma

from get_embedding_function import get_embedding_function
from rag_config import load_config, project_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    config = load_config()

    if args.reset:
        print("Clearing Chroma database")
        clear_database(config)

    documents = load_documents(config)
    chunks = split_documents(documents, config)
    add_to_chroma(chunks, config)


def load_documents(config: dict | None = None) -> list[Document]:
    config = config or load_config()
    papers_dir = project_path(config["paths"]["papers_dir"])
    papers_dir.mkdir(parents=True, exist_ok=True)

    document_loader = PyPDFDirectoryLoader(str(papers_dir))
    documents = document_loader.load()
    return [normalize_document(document) for document in documents]


def normalize_document(document: Document) -> Document:
    source_path = Path(document.metadata.get("source", "unknown.pdf"))
    page_index = int(document.metadata.get("page", 0))
    source_file = source_path.name

    metadata = {
        **document.metadata,
        "source_file": source_file,
        "page_number": page_index + 1,
        "paper_id": make_paper_id(source_file),
    }

    return Document(page_content=clean_text(document.page_content), metadata=metadata)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_paper_id(source_file: str) -> str:
    stem = Path(source_file).stem.lower()
    return re.sub(r"[^a-z0-9]+", "-", stem).strip("-")


def split_documents(
    documents: list[Document], config: dict | None = None
) -> list[Document]:
    config = config or load_config()
    chunk_config = config["chunking"]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_config["chunk_size"]),
        chunk_overlap=int(chunk_config["chunk_overlap"]),
        length_function=len,
        is_separator_regex=False,
    )
    return calculate_chunk_ids(text_splitter.split_documents(documents))


def add_to_chroma(chunks: list[Document], config: dict | None = None):
    config = config or load_config()
    chroma_path = project_path(config["paths"]["chroma_dir"])
    chroma_path.mkdir(parents=True, exist_ok=True)

    db = Chroma(
        persist_directory=str(chroma_path),
        embedding_function=get_embedding_function(config),
    )

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks:
        if chunk.metadata["chunk_id"] not in existing_ids:
            new_chunks.append(chunk)

    if new_chunks:
        print(f"Adding new chunks: {len(new_chunks)}")
        db.add_documents(
            new_chunks,
            ids=[chunk.metadata["chunk_id"] for chunk in new_chunks],
        )
        if hasattr(db, "persist"):
            db.persist()
    else:
        print("No new chunks to add")


def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source_file = chunk.metadata["source_file"]
        page_number = chunk.metadata["page_number"]
        paper_id = chunk.metadata["paper_id"]
        current_page_id = f"{paper_id}:{page_number}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{paper_id}:p{page_number}:c{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata["id"] = chunk_id
        chunk.metadata["source_file"] = source_file
        chunk.metadata["page_number"] = page_number

    return chunks


def clear_database(config: dict | None = None):
    config = config or load_config()
    chroma_path = project_path(config["paths"]["chroma_dir"])
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)


if __name__ == "__main__":
    main()
