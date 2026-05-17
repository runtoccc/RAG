from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_config import load_config, project_path


CHUNK_STYLE = "openscholar_256w_title_prefix"


def main() -> None:
    configure_stdout()
    args = parse_args()
    config = load_config()
    passages_path = project_path(
        args.input or config["paths"].get("passages_jsonl", "data/passages/scientific_passages.jsonl")
    )
    passages = read_jsonl(passages_path)
    report = {
        "path": str(passages_path),
        "stats": calculate_passage_quality_stats(passages),
        "warnings": build_warnings(passages),
        "examples": sample_passages(passages, count=3),
    }
    report_json = Path(args.report_json)
    report_md = Path(args.report_md)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check OpenScholar-style passage quality.")
    parser.add_argument("--input", default=None)
    parser.add_argument("--report-json", default="outputs/reports/passage_quality_report.json")
    parser.add_argument("--report-md", default="outputs/reports/passage_quality_report.md")
    return parser.parse_args()


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Passages file does not exist: {path}")
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def calculate_passage_quality_stats(passages: list[dict[str, Any]]) -> dict[str, Any]:
    total_passages = len(passages)
    block_words = [int(passage.get("block_words") or 0) for passage in passages]
    paper_counts = Counter(passage.get("paper_id") for passage in passages)
    title_prefix_valid_count = sum(
        1 for passage in passages if has_valid_title_prefix(passage)
    )
    has_section_in_embedding_text_count = sum(
        1 for passage in passages if "Section:" in str(passage.get("embedding_text") or "")
    )
    section_label_leak_count = sum(1 for passage in passages if has_label_leak(passage))
    count_blocks_eq_256 = sum(1 for count in block_words if count == 256)
    count_blocks_lt_64 = sum(1 for count in block_words if count < 64)
    count_blocks_gt_256 = sum(1 for count in block_words if count > 256)
    count_blocks_gt_320 = sum(1 for count in block_words if count > 320)
    chunk_style_distribution = Counter(
        passage.get("chunk_style") or "missing" for passage in passages
    )

    return {
        "total_passages": total_passages,
        "avg_block_words": statistics.mean(block_words) if block_words else 0.0,
        "min_block_words": min(block_words) if block_words else 0,
        "max_block_words": max(block_words) if block_words else 0,
        "count_blocks_eq_256": count_blocks_eq_256,
        "ratio_blocks_eq_256": count_blocks_eq_256 / total_passages if total_passages else 0.0,
        "count_blocks_lt_64": count_blocks_lt_64,
        "count_blocks_gt_256": count_blocks_gt_256,
        "count_blocks_gt_320": count_blocks_gt_320,
        "title_prefix_valid_count": title_prefix_valid_count,
        "title_prefix_invalid_count": total_passages - title_prefix_valid_count,
        "has_section_in_embedding_text_count": has_section_in_embedding_text_count,
        "section_label_leak_count": section_label_leak_count,
        "chunk_style_distribution": dict(sorted(chunk_style_distribution.items())),
        "unique_paper_count": len([paper_id for paper_id in paper_counts if paper_id]),
        "avg_blocks_per_paper": statistics.mean(paper_counts.values()) if paper_counts else 0.0,
        "bad_title_count": count_bad_titles(passages),
        "missing_title_count": count_missing(passages, "title"),
        "missing_source_file_count": count_missing(passages, "source_file"),
        "missing_main_text_count": sum(1 for passage in passages if passage.get("missing_main_text")),
        "not_index_ready_passage_count": sum(
            1 for passage in passages if passage.get("index_ready") is False
        ),
        "bad_title_in_passage_count": count_flagged_passages(passages, "bad_title"),
        "bad_fallback_title_in_passage_count": count_flagged_passages(
            passages, "bad_fallback_title"
        ),
        "bad_passage_id_count": sum(
            1 for passage in passages if "::block::" not in str(passage.get("passage_id") or "")
        ),
        "bad_chunk_style_count": sum(
            1 for passage in passages if passage.get("chunk_style") != CHUNK_STYLE
        ),
        "bad_embedding_text_count": total_passages - title_prefix_valid_count,
        "unexpected_small_block_count": count_unexpected_small_blocks(passages),
    }


def has_valid_title_prefix(passage: dict[str, Any]) -> bool:
    title = passage.get("title") or ""
    text = passage.get("text") or ""
    return passage.get("embedding_text") == f"{title}\n\n{text}"


def has_label_leak(passage: dict[str, Any]) -> bool:
    embedding_text = str(passage.get("embedding_text") or "")
    return any(label in embedding_text for label in ["Section:", "Paper title:", "Passage:"])


def count_bad_titles(passages: list[dict[str, Any]]) -> int:
    count = 0
    for passage in passages:
        title = str(passage.get("title") or "").strip()
        if not title or len(title) > 220:
            count += 1
    return count


def count_missing(passages: list[dict[str, Any]], key: str) -> int:
    return sum(1 for passage in passages if passage.get(key) in (None, ""))


def count_flagged_passages(passages: list[dict[str, Any]], flag: str) -> int:
    return sum(1 for passage in passages if flag in (passage.get("quality_flags") or []))


def count_unexpected_small_blocks(passages: list[dict[str, Any]]) -> int:
    by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for passage in passages:
        by_paper[str(passage.get("paper_id") or "")].append(passage)

    count = 0
    for paper_passages in by_paper.values():
        total_words = sum(int(passage.get("block_words") or 0) for passage in paper_passages)
        for passage in paper_passages:
            block_word_count = int(passage.get("block_words") or 0)
            flags = passage.get("quality_flags") or []
            allowed_too_short = (
                total_words < 64 and "too_short_for_standard_block" in flags
            )
            if block_word_count < 64 and not allowed_too_short:
                count += 1
    return count


def build_warnings(passages: list[dict[str, Any]]) -> list[str]:
    stats = calculate_passage_quality_stats(passages)
    warnings = []
    if stats["bad_embedding_text_count"]:
        warnings.append("WARNING: some embedding_text values are not title + blank line + text")
    if stats["section_label_leak_count"]:
        warnings.append("WARNING: embedding_text contains Section:, Paper title:, or Passage:")
    if stats["bad_passage_id_count"]:
        warnings.append("WARNING: some passage_id values do not contain ::block::")
    if stats["bad_chunk_style_count"]:
        warnings.append(f"WARNING: some chunk_style values are not {CHUNK_STYLE}")
    if stats["unexpected_small_block_count"]:
        warnings.append("WARNING: some blocks are <64 words without a whole-paper too-short flag")
    if stats["missing_title_count"]:
        warnings.append("WARNING: some passages are missing title")
    if stats["missing_source_file_count"]:
        warnings.append("WARNING: some passages are missing source_file")
    if stats["not_index_ready_passage_count"]:
        warnings.append("WARNING: some passages were built from non-index-ready records")
    if stats["bad_fallback_title_in_passage_count"]:
        warnings.append("WARNING: some passages contain bad fallback titles")
    return warnings


def build_markdown(report: dict[str, Any]) -> str:
    stats = report["stats"]
    lines = [
        "# Passage Quality Report",
        "",
        f"- total_passages: {stats['total_passages']}",
        f"- unique_paper_count: {stats['unique_paper_count']}",
        f"- avg_blocks_per_paper: {stats['avg_blocks_per_paper']:.2f}",
        f"- avg_block_words: {stats['avg_block_words']:.2f}",
        f"- min_block_words: {stats['min_block_words']}",
        f"- max_block_words: {stats['max_block_words']}",
        f"- count_blocks_eq_256: {stats['count_blocks_eq_256']}",
        f"- ratio_blocks_eq_256: {stats['ratio_blocks_eq_256']:.3f}",
        f"- count_blocks_lt_64: {stats['count_blocks_lt_64']}",
        f"- count_blocks_gt_320: {stats['count_blocks_gt_320']}",
        f"- title_prefix_valid_count: {stats['title_prefix_valid_count']}",
        f"- title_prefix_invalid_count: {stats['title_prefix_invalid_count']}",
        f"- section_label_leak_count: {stats['section_label_leak_count']}",
        f"- missing_main_text_count: {stats['missing_main_text_count']}",
        f"- not_index_ready_passage_count: {stats['not_index_ready_passage_count']}",
        f"- bad_fallback_title_in_passage_count: {stats['bad_fallback_title_in_passage_count']}",
        "",
        "## Warnings",
        json.dumps(report["warnings"], ensure_ascii=False, indent=2),
    ]
    return "\n".join(lines) + "\n"


def sample_passages(passages: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    sampled = random.sample(passages, k=min(count, len(passages)))
    return [
        {
            "passage_id": passage.get("passage_id"),
            "paper_id": passage.get("paper_id"),
            "block_index": passage.get("block_index"),
            "block_words": passage.get("block_words"),
            "embedding_text_preview": make_preview(passage.get("embedding_text") or ""),
        }
        for passage in sampled
    ]


def make_preview(text: str, max_chars: int = 900) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


if __name__ == "__main__":
    main()
