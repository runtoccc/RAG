from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path


def main() -> None:
    args = parse_args()
    papers_dir = Path(args.papers_dir)
    tei_dir = Path(args.tei_dir)
    clean_path = Path(args.clean_jsonl)
    passages_path = Path(args.passages_jsonl)
    failures_path = Path(args.failures_csv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_count = len(list(papers_dir.glob("*.pdf")))
    tei_count = len(list(tei_dir.glob("*.tei.xml")))
    failure_count = count_parse_failures(failures_path)
    clean_records = read_jsonl(clean_path)
    passage_records = read_jsonl(passages_path)

    stats = build_stats(
        pdf_count=pdf_count,
        tei_count=tei_count,
        failure_count=failure_count,
        clean_records=clean_records,
        passage_records=passage_records,
    )
    output_path.write_text(render_report(stats), encoding="utf-8")
    print(f"[report] output={output_path}")
    print(
        f"[report] pdf={stats['pdf_count']} parsed={stats['parsed_count']} "
        f"failed={stats['failed_count']} clean={stats['clean_text_count']} "
        f"passages={stats['passage_count']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PDF processing quality report.")
    parser.add_argument("--papers-dir", default="data/papers")
    parser.add_argument("--tei-dir", default="data/interim/tei")
    parser.add_argument("--clean-jsonl", default="data/clean/pes2o_like.jsonl")
    parser.add_argument("--passages-jsonl", default="data/passages/scientific_passages.jsonl")
    parser.add_argument("--failures-csv", default="outputs/reports/parse_failures.csv")
    parser.add_argument("--output", default="outputs/reports/pdf_processing_quality_report.md")
    return parser.parse_args()


def build_stats(
    pdf_count: int,
    tei_count: int,
    failure_count: int,
    clean_records: list[dict],
    passage_records: list[dict],
) -> dict:
    word_counts = [int((record.get("metadata") or {}).get("n_words") or 0) for record in clean_records]
    paragraph_counts = [
        int((record.get("metadata") or {}).get("n_paragraphs") or 0)
        for record in clean_records
    ]
    section_distribution = Counter(
        passage.get("section_type") or "unknown" for passage in passage_records
    )
    removed_sections = Counter()
    for record in clean_records:
        for section in (record.get("metadata") or {}).get("removed_sections") or []:
            removed_sections[section or "Unknown"] += 1

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pdf_count": pdf_count,
        "parsed_count": tei_count,
        "failed_count": failure_count,
        "clean_text_count": len(clean_records),
        "passage_count": len(passage_records),
        "avg_words": safe_average(word_counts),
        "avg_paragraphs": safe_average(paragraph_counts),
        "section_distribution": dict(sorted(section_distribution.items())),
        "removed_section_distribution": dict(sorted(removed_sections.items())),
    }


def count_parse_failures(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return sum(1 for row in reader if row.get("source_file"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def safe_average(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def render_report(stats: dict) -> str:
    lines = [
        "# PDF Processing Quality Report",
        "",
        f"Generated at: `{stats['generated_at']}`",
        "",
        "## Summary",
        "",
        f"- PDF total: `{stats['pdf_count']}`",
        f"- Successfully parsed TEI files: `{stats['parsed_count']}`",
        f"- Parse failures: `{stats['failed_count']}`",
        f"- Clean text records: `{stats['clean_text_count']}`",
        f"- Scientific passages: `{stats['passage_count']}`",
        f"- Average words per clean text: `{stats['avg_words']:.2f}`",
        f"- Average paragraphs per clean text: `{stats['avg_paragraphs']:.2f}`",
        "",
        "## Section Distribution",
        "",
        "| Section | Count |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {section} | {count} |"
        for section, count in stats["section_distribution"].items()
    )
    lines.extend(
        [
            "",
            "## Removed Section Types",
            "",
            "| Removed section | Count |",
            "| --- | ---: |",
        ]
    )
    if stats["removed_section_distribution"]:
        lines.extend(
            f"| {section} | {count} |"
            for section, count in stats["removed_section_distribution"].items()
        )
    else:
        lines.append("| None | 0 |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
