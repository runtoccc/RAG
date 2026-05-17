from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_STRUCTURED = "data/structured/local_s2orc_enriched.jsonl"
DEFAULT_CLEAN_PASS = "data/clean/pes2o_style_pass.jsonl"
DEFAULT_PASSAGES = "data/passages/openscholar_passages.jsonl"
DEFAULT_OUTPUT = "data/agent/paper_registry.jsonl"


def main() -> None:
    args = parse_args()
    structured_path = Path(args.structured)
    clean_pass_path = Path(args.clean_pass)
    passages_path = Path(args.passages)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pass_records = load_clean_pass_records(clean_pass_path)
    passage_counts = count_passages(passages_path)

    total = 0
    index_ready_count = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in iter_jsonl(structured_path):
            total += 1
            paper_id = record.get("paper_id") or ""
            metadata = record.get("metadata") or {}
            pass_metadata = pass_records.get(paper_id, {})
            index_ready = bool(pass_metadata.get("index_ready"))
            if index_ready:
                index_ready_count += 1
            source_file = record.get("source_file") or metadata.get("source_file") or ""
            row = {
                "paper_id": paper_id,
                "source_file": source_file,
                "pdf_sha256": sha256_file(Path(args.pdf_dir) / source_file) if source_file else None,
                "doi": metadata.get("doi"),
                "title": record.get("title") or metadata.get("title"),
                "year": metadata.get("year"),
                "parse_status": record.get("parse_status"),
                "index_ready": index_ready,
                "passage_count": passage_counts.get(paper_id, 0),
                "body_paragraph_count": len(record.get("body_text") or []),
                "bib_entry_count": len(record.get("bib_entries") or {}),
                "cite_span_count": count_nested_spans(record.get("body_text") or [], "cite_spans"),
                "ref_span_count": count_nested_spans(record.get("body_text") or [], "ref_spans"),
                "quality_flags": unique_keep_order(
                    (record.get("quality_flags") or []) + (pass_metadata.get("quality_flags") or [])
                ),
            }
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[paper-registry] structured={structured_path}")
    print(f"[paper-registry] total_papers={total}")
    print(f"[paper-registry] index_ready_papers={index_ready_count}")
    print(f"[paper-registry] output={output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a traceable paper registry for the local literature agent.")
    parser.add_argument("--structured", default=DEFAULT_STRUCTURED)
    parser.add_argument("--clean-pass", default=DEFAULT_CLEAN_PASS)
    parser.add_argument("--passages", default=DEFAULT_PASSAGES)
    parser.add_argument("--pdf-dir", default="data/papers")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def iter_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_clean_pass_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    for record in iter_jsonl(path):
        paper_id = record.get("paper_id")
        metadata = record.get("metadata") or {}
        if paper_id:
            records[paper_id] = {
                "index_ready": metadata.get("index_ready") is True,
                "quality_flags": metadata.get("quality_flags") or [],
            }
    return records


def count_passages(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    for passage in iter_jsonl(path):
        paper_id = passage.get("paper_id")
        if paper_id:
            counts[paper_id] = counts.get(paper_id, 0) + 1
    return counts


def count_nested_spans(paragraphs: list[dict[str, Any]], field: str) -> int:
    return sum(len(paragraph.get(field) or []) for paragraph in paragraphs)


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_keep_order(values: list[Any]) -> list[Any]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


if __name__ == "__main__":
    main()
