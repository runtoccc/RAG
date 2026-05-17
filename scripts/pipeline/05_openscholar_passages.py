from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_BLOCK_WORDS = 256
CHUNK_STYLE = "openscholar_256w_title_prefix"
ALLOWED_INPUT_NAMES = {"pes2o_style_pass.jsonl", "pes2o_like_pass.jsonl"}
FORBIDDEN_INPUT_NAMES = {"pes2o_style.jsonl", "pes2o_like.jsonl"}


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    compat_output = Path(args.compat_output) if args.compat_output else None
    validate_input_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if compat_output:
        compat_output.parent.mkdir(parents=True, exist_ok=True)

    stats = new_stats()
    with output_path.open("w", encoding="utf-8") as output_file:
        compat_file = compat_output.open("w", encoding="utf-8") if compat_output else None
        try:
            for record in iter_jsonl(input_path):
                stats["records"] += 1
                for passage in build_passages_for_record(record, strict=args.strict):
                    validate_passage(passage)
                    update_stats(stats, passage)
                    line = json.dumps(passage, ensure_ascii=False) + "\n"
                    output_file.write(line)
                    if compat_file:
                        compat_file.write(line)
        finally:
            if compat_file:
                compat_file.close()
    assert_output_stats(stats)

    print(f"[openscholar-passages] input={input_path}")
    print(f"[openscholar-passages] records={stats['records']}")
    print(f"[openscholar-passages] output={output_path}")
    if compat_output:
        print(f"[openscholar-passages] compat_output={compat_output}")
    print(f"[openscholar-passages] total_passages={stats['total_passages']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local OpenScholar-style title-prefixed 256-word passages.")
    parser.add_argument("--input", default="data/clean/pes2o_style_pass.jsonl")
    parser.add_argument("--output", default="data/passages/openscholar_passages.jsonl")
    parser.add_argument("--compat-output", default="data/passages/scientific_passages.jsonl")
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    return parser.parse_args()


def validate_input_path(path: Path) -> None:
    if path.name in FORBIDDEN_INPUT_NAMES:
        raise RuntimeError(
            "Refusing to build passages from an all-record clean file. Use pes2o_style_pass.jsonl."
        )
    if path.name not in ALLOWED_INPUT_NAMES:
        raise RuntimeError(f"Strict passage building requires a pass file, got: {path}")


def build_passages_for_record(record: dict[str, Any], strict: bool = True) -> list[dict[str, Any]]:
    metadata = record.get("metadata") or {}
    paper_id = record.get("paper_id") or "unknown_paper"
    if metadata.get("index_ready") is not True:
        message = f"Record is not index_ready and cannot be chunked: {paper_id}"
        if strict:
            raise RuntimeError(message)
        print(f"WARNING: {message}")
        return []

    main_text = record.get("main_text") or ""
    if not main_text:
        message = f"main_text is required for OpenScholar-style passages: {paper_id}"
        if strict:
            raise RuntimeError(message)
        print(f"WARNING: {message}")
        return []

    title = normalize_text(record.get("title"))
    if not title:
        raise RuntimeError(f"Missing title for index_ready record: {paper_id}")
    source_file = normalize_text(metadata.get("source_file"))
    if not source_file:
        raise RuntimeError(f"Missing source_file for index_ready record: {paper_id}")

    passages = []
    for block_index, block_words in enumerate(split_into_blocks(main_text)):
        block_text = " ".join(block_words)
        passage = {
            "passage_id": f"{paper_id}::block::{block_index:04d}",
            "paper_id": paper_id,
            "title": title,
            "block_index": block_index,
            "block_words": len(block_words),
            "text": block_text,
            "embedding_text": f"{title}\n\n{block_text}",
            "source_file": source_file,
            "doi": metadata.get("doi"),
            "source": record.get("source") or "local_grobid_s2orc_style",
            "chunk_style": CHUNK_STYLE,
            "quality_flags": metadata.get("quality_flags") or [],
            "index_ready": metadata.get("index_ready"),
            "pass_pes2o_filters": metadata.get("pass_pes2o_filters"),
            "pass_hard_filters": metadata.get("pass_hard_filters"),
            "pass_quality_audit": metadata.get("pass_quality_audit"),
            "fail_reasons": metadata.get("fail_reasons") or [],
            "source_text_field": "main_text",
        }
        validate_passage(passage)
        passages.append(passage)
    return passages


def split_into_blocks(text: str) -> list[list[str]]:
    words = text.split()
    return [words[start : start + DEFAULT_BLOCK_WORDS] for start in range(0, len(words), DEFAULT_BLOCK_WORDS) if words[start : start + DEFAULT_BLOCK_WORDS]]


def validate_passage(passage: dict[str, Any]) -> None:
    if int(passage.get("block_words") or 0) > DEFAULT_BLOCK_WORDS:
        raise RuntimeError(f"Passage block is too large: {passage.get('passage_id')}")
    if passage.get("embedding_text") != f"{passage.get('title')}\n\n{passage.get('text')}":
        raise RuntimeError(f"Bad embedding_text format: {passage.get('passage_id')}")
    if any(label in str(passage.get("embedding_text") or "") for label in ["Section:", "Paper title:", "Passage:"]):
        raise RuntimeError(f"Label leakage in embedding_text: {passage.get('passage_id')}")
    if passage.get("index_ready") is not True:
        raise RuntimeError(f"Non-index-ready passage leaked: {passage.get('passage_id')}")
    if passage.get("pass_pes2o_filters") is not True:
        raise RuntimeError(f"Non-pass peS2o record leaked: {passage.get('passage_id')}")
    if passage.get("fail_reasons"):
        raise RuntimeError(f"Passage carries fail_reasons: {passage.get('passage_id')}")
    if not passage.get("title") or not passage.get("source_file"):
        raise RuntimeError(f"Missing title/source_file in passage: {passage.get('passage_id')}")


def new_stats() -> dict[str, int]:
    return {
        "records": 0,
        "total_passages": 0,
        "count_blocks_gt_256": 0,
        "bad_embedding_text_count": 0,
        "not_index_ready_passage_count": 0,
        "missing_main_text_count": 0,
        "label_leak_count": 0,
    }


def update_stats(stats: dict[str, int], passage: dict[str, Any]) -> None:
    stats["total_passages"] += 1
    if int(passage.get("block_words") or 0) > DEFAULT_BLOCK_WORDS:
        stats["count_blocks_gt_256"] += 1
    if passage.get("embedding_text") != f"{passage.get('title')}\n\n{passage.get('text')}":
        stats["bad_embedding_text_count"] += 1
    if passage.get("index_ready") is not True:
        stats["not_index_ready_passage_count"] += 1
    if passage.get("source_text_field") != "main_text":
        stats["missing_main_text_count"] += 1
    if any(label in str(passage.get("embedding_text") or "") for label in ["Section:", "Paper title:", "Passage:"]):
        stats["label_leak_count"] += 1


def assert_output_stats(stats: dict[str, int]) -> None:
    if stats["total_passages"] <= 0:
        raise RuntimeError("No passages were generated.")
    for key in [
        "count_blocks_gt_256",
        "bad_embedding_text_count",
        "not_index_ready_passage_count",
        "missing_main_text_count",
        "label_leak_count",
    ]:
        if stats[key] != 0:
            raise RuntimeError(f"OpenScholar passage assertion failed: {key}={stats[key]}")


def iter_jsonl(path: Path):
    if not path.exists():
        raise RuntimeError(f"Input file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


if __name__ == "__main__":
    main()
