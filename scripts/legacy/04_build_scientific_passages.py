from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_BLOCK_WORDS = 256
DEFAULT_MIN_FINAL_WORDS = 64
CHUNK_STYLE = "openscholar_256w_title_prefix"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(input_path)
    warn_if_all_records_input(input_path)
    print(f"[passages] input={input_path}")
    print(f"[passages] records={len(records)}")
    print(f"[passages] output={output_path}")
    print(f"[passages] block_words={args.block_words} min_final_words={args.min_final_words}")

    total_passages = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            passages = build_passages_for_record(
                record,
                block_words=args.block_words,
                min_final_words=args.min_final_words,
            )
            for passage in passages:
                output_file.write(json.dumps(passage, ensure_ascii=False) + "\n")
            total_passages += len(passages)
            print(f"[ok] {record.get('paper_id')} passages={len(passages)}")

    print(f"[done] total_passages={total_passages}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build OpenScholar-style 256-word title-prefixed passages from peS2o-like JSONL."
    )
    parser.add_argument("--input", default="data/clean/pes2o_like_pass.jsonl")
    parser.add_argument("--output", default="data/passages/scientific_passages.jsonl")
    parser.add_argument("--block-words", type=int, default=DEFAULT_BLOCK_WORDS)
    parser.add_argument("--min-final-words", type=int, default=DEFAULT_MIN_FINAL_WORDS)
    return parser.parse_args()


def warn_if_all_records_input(input_path: Path) -> None:
    if input_path.name == "pes2o_like.jsonl":
        print(
            "WARNING: You are building OpenScholar-style passages from all records, "
            "including failed records. Prefer data/clean/pes2o_like_pass.jsonl."
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_passages_for_record(
    record: dict[str, Any],
    block_words: int = DEFAULT_BLOCK_WORDS,
    min_final_words: int = DEFAULT_MIN_FINAL_WORDS,
) -> list[dict[str, Any]]:
    paper_id = record.get("paper_id") or "unknown_paper"
    title = normalize_metadata_text(record.get("title")) or "Unknown title"
    source = record.get("source") or "local_grobid"
    metadata = record.get("metadata") or {}
    if metadata.get("index_ready") is False:
        print(f"WARNING: Skipping non-index-ready record: {paper_id}")
        return []
    source_file = metadata.get("source_file")
    doi = metadata.get("doi")
    parse_status = metadata.get("parse_status")
    inherited_quality_flags = list(metadata.get("quality_flags") or [])
    index_ready = metadata.get("index_ready")
    pass_hard_filters = metadata.get("pass_hard_filters")
    pass_quality_audit = metadata.get("pass_quality_audit")
    source_text = record.get("main_text") or ""
    missing_main_text = False
    if not source_text:
        missing_main_text = True
        source_text = record.get("text") or ""
        print(f"WARNING: main_text missing for {paper_id}; falling back to record['text'].")
    blocks = split_text_into_word_blocks(
        source_text,
        block_words=block_words,
        min_final_words=min_final_words,
    )

    passages = []
    for block_index, block in enumerate(blocks):
        block_quality_flags = list(inherited_quality_flags)
        if block["is_too_short_for_standard_block"]:
            block_quality_flags.append("too_short_for_standard_block")

        text = block["text"]
        passage_id = f"{paper_id}::block::{block_index:04d}"
        passages.append(
            {
                "passage_id": passage_id,
                "paper_id": paper_id,
                "title": title,
                "block_index": block_index,
                "block_words": block["block_words"],
                "text": text,
                "embedding_text": f"{title}\n\n{text}",
                "source_file": source_file,
                "page": None,
                "doi": doi,
                "source": source,
                "chunk_style": CHUNK_STYLE,
                "quality_flags": unique_keep_order(block_quality_flags),
                "parse_status": parse_status,
                "index_ready": index_ready,
                "pass_hard_filters": pass_hard_filters,
                "pass_quality_audit": pass_quality_audit,
                "missing_main_text": missing_main_text,
                "section_title": None,
                "section_type": None,
            }
        )
    return passages


def split_text_into_word_blocks(
    text: str,
    block_words: int = DEFAULT_BLOCK_WORDS,
    min_final_words: int = DEFAULT_MIN_FINAL_WORDS,
) -> list[dict[str, Any]]:
    words = text.split()
    if len(words) < min_final_words:
        return [
            {
                "text": " ".join(words),
                "block_words": len(words),
                "is_too_short_for_standard_block": True,
            }
        ]

    blocks: list[list[str]] = []
    cursor = 0
    while cursor + block_words <= len(words):
        blocks.append(words[cursor : cursor + block_words])
        cursor += block_words

    remaining = words[cursor:]
    if remaining:
        if len(remaining) >= min_final_words or not blocks:
            blocks.append(remaining)
        else:
            blocks[-1].extend(remaining)

    return [
        {
            "text": " ".join(block),
            "block_words": len(block),
            "is_too_short_for_standard_block": False,
        }
        for block in blocks
    ]


def normalize_metadata_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def unique_keep_order(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique


if __name__ == "__main__":
    main()
