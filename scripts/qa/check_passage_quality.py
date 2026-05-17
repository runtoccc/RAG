from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def main() -> None:
    args = parse_args()
    passages = read_jsonl(Path(args.input))
    stats = build_stats(passages)
    write_report(Path(args.report_json), Path(args.report_md), stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local OpenScholar-style passage quality.")
    parser.add_argument("--input", default="data/passages/openscholar_passages.jsonl")
    parser.add_argument("--report-json", default="outputs/reports/openscholar_conformance_report.json")
    parser.add_argument("--report-md", default="outputs/reports/openscholar_conformance_report.md")
    return parser.parse_args()


def build_stats(passages: list[dict]) -> dict:
    block_words = [int(p.get("block_words") or 0) for p in passages]
    return {
        "total_passages": len(passages),
        "unique_paper_count": len({p.get("paper_id") for p in passages if p.get("paper_id")}),
        "all_block_words_lte_256": all(count <= 256 for count in block_words),
        "count_blocks_eq_256": sum(1 for count in block_words if count == 256),
        "count_blocks_lt_256": sum(1 for count in block_words if count < 256),
        "count_blocks_gt_256": sum(1 for count in block_words if count > 256),
        "title_prefix_valid_count": sum(1 for p in passages if valid_title_prefix(p)),
        "label_leak_count": sum(1 for p in passages if label_leak(p)),
        "not_index_ready_passage_count": sum(1 for p in passages if p.get("index_ready") is not True),
        "missing_main_text_count": sum(1 for p in passages if p.get("missing_main_text")),
        "bad_embedding_text_count": sum(1 for p in passages if not valid_title_prefix(p)),
        "source_text_is_main_text_count": sum(
            1 for p in passages if p.get("source_text_field") == "main_text"
        ),
        "chunk_style_distribution": dict(Counter(p.get("chunk_style") or "missing" for p in passages).most_common()),
    }


def valid_title_prefix(passage: dict) -> bool:
    return passage.get("embedding_text") == f"{passage.get('title')}\n\n{passage.get('text')}"


def label_leak(passage: dict) -> bool:
    text = str(passage.get("embedding_text") or "")
    return any(label in text for label in ["Section:", "Paper title:", "Passage:"])


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_report(json_path: Path, md_path: Path, stats: dict) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("# OpenScholar-style Passage Quality Report\n\n```json\n" + json.dumps(stats, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")


if __name__ == "__main__":
    main()
