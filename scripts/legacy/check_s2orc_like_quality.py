from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    report_md = Path(args.report_md)
    report_json = Path(args.report_json)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(input_path)
    stats = build_stats(records)
    report_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(build_markdown(stats), encoding="utf-8")

    print(f"[s2orc-quality] input={input_path}")
    print(f"[s2orc-quality] total_records={stats['total_records']}")
    print(f"[s2orc-quality] total_body_text_count={stats['total_body_text_count']}")
    print(f"[s2orc-quality] total_bib_entries_count={stats['total_bib_entries_count']}")
    print(f"[s2orc-quality] report_md={report_md}")
    print(f"[s2orc-quality] report_json={report_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local S2ORC-like structured JSONL quality.")
    parser.add_argument("--input", default="data/structured/s2orc_like.jsonl")
    parser.add_argument("--report-md", default="outputs/reports/s2orc_like_quality_report.md")
    parser.add_argument("--report-json", default="outputs/reports/s2orc_like_quality_report.json")
    return parser.parse_args()


def build_stats(records: list[dict]) -> dict:
    body_counts = [len(record.get("body_text") or []) for record in records]
    bib_counts = [len(record.get("bib_entries") or {}) for record in records]
    quality_counter = Counter(
        flag for record in records for flag in record.get("quality_flags") or []
    )
    parse_status_counter = Counter(record.get("parse_status") or "unknown" for record in records)

    return {
        "total_records": len(records),
        "missing_title_count": count_missing(records, "title"),
        "missing_abstract_count": count_missing(records, "abstract"),
        "missing_body_count": sum(1 for count in body_counts if count == 0),
        "missing_doi_count": sum(
            1 for record in records if not (record.get("metadata") or {}).get("doi")
        ),
        "missing_year_count": sum(
            1 for record in records if not (record.get("metadata") or {}).get("year")
        ),
        "avg_body_paragraphs": average(body_counts),
        "min_body_paragraphs": min(body_counts) if body_counts else 0,
        "max_body_paragraphs": max(body_counts) if body_counts else 0,
        "total_body_text_count": sum(body_counts),
        "total_bib_entries_count": sum(bib_counts),
        "avg_bib_entries_per_paper": average(bib_counts),
        "parse_status_distribution": dict(parse_status_counter.most_common()),
        "quality_flags_distribution": dict(quality_counter.most_common()),
        "examples_missing_title": examples(records, lambda record: not record.get("title")),
        "examples_missing_abstract": examples(records, lambda record: not record.get("abstract")),
        "examples_empty_body": examples(records, lambda record: not record.get("body_text")),
        "samples": sample_records(records),
    }


def build_markdown(stats: dict) -> str:
    lines = [
        "# S2ORC-like Quality Report",
        "",
        f"- total_records: {stats['total_records']}",
        f"- missing_title_count: {stats['missing_title_count']}",
        f"- missing_abstract_count: {stats['missing_abstract_count']}",
        f"- missing_body_count: {stats['missing_body_count']}",
        f"- missing_doi_count: {stats['missing_doi_count']}",
        f"- missing_year_count: {stats['missing_year_count']}",
        f"- avg_body_paragraphs: {stats['avg_body_paragraphs']:.2f}",
        f"- min_body_paragraphs: {stats['min_body_paragraphs']}",
        f"- max_body_paragraphs: {stats['max_body_paragraphs']}",
        f"- total_body_text_count: {stats['total_body_text_count']}",
        f"- total_bib_entries_count: {stats['total_bib_entries_count']}",
        f"- avg_bib_entries_per_paper: {stats['avg_bib_entries_per_paper']:.2f}",
        "",
        "## Parse Status Distribution",
        json.dumps(stats["parse_status_distribution"], ensure_ascii=False, indent=2),
        "",
        "## Quality Flags Distribution",
        json.dumps(stats["quality_flags_distribution"], ensure_ascii=False, indent=2),
        "",
        "## Samples",
    ]
    for sample in stats["samples"]:
        lines.extend(
            [
                "",
                f"### {sample['paper_id']}",
                f"- source_file: {sample['source_file']}",
                f"- title: {sample['title']}",
                f"- abstract: {sample['abstract_preview']}",
                "- body_text:",
            ]
        )
        for paragraph in sample["body_text_preview"]:
            lines.append(f"  - {paragraph}")
    return "\n".join(lines) + "\n"


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


def count_missing(records: list[dict], field: str) -> int:
    return sum(1 for record in records if not record.get(field))


def average(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def examples(records: list[dict], predicate, limit: int = 5) -> list[dict]:
    selected = []
    for record in records:
        if predicate(record):
            selected.append(
                {
                    "paper_id": record.get("paper_id"),
                    "source_file": record.get("source_file"),
                    "title": record.get("title"),
                }
            )
        if len(selected) >= limit:
            break
    return selected


def sample_records(records: list[dict], limit: int = 5) -> list[dict]:
    samples = []
    for record in records[:limit]:
        body_preview = []
        for paragraph in (record.get("body_text") or [])[:3]:
            text = paragraph.get("text") if isinstance(paragraph, dict) else str(paragraph)
            body_preview.append(preview(text, 240))
        samples.append(
            {
                "paper_id": record.get("paper_id"),
                "source_file": record.get("source_file"),
                "title": record.get("title"),
                "abstract_preview": preview(record.get("abstract") or "", 300),
                "body_text_preview": body_preview,
            }
        )
    return samples


def preview(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


if __name__ == "__main__":
    main()
