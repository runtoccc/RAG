from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def main() -> None:
    args = parse_args()
    records = read_jsonl(Path(args.input))
    stats = build_stats(records)
    write_report(Path(args.report_json), Path(args.report_md), stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local S2ORC-aligned structured JSONL quality.")
    parser.add_argument("--input", default="data/structured/local_s2orc.jsonl")
    parser.add_argument("--report-json", default="outputs/reports/s2orc_conformance_report.json")
    parser.add_argument("--report-md", default="outputs/reports/s2orc_conformance_report.md")
    return parser.parse_args()


def build_stats(records: list[dict]) -> dict:
    parse_status = Counter(record.get("parse_status") or "unknown" for record in records)
    flags = Counter(flag for record in records for flag in record.get("quality_flags") or [])
    cite_spans = [
        span
        for record in records
        for paragraph in record.get("body_text") or []
        for span in paragraph.get("cite_spans") or []
    ]
    bib_keys_by_record = {
        record.get("paper_id"): set((record.get("bib_entries") or {}).keys()) for record in records
    }
    linked_cite_spans = 0
    unresolved_cite_spans = 0
    for record in records:
        bib_keys = bib_keys_by_record.get(record.get("paper_id"), set())
        for paragraph in record.get("body_text") or []:
            for span in paragraph.get("cite_spans") or []:
                if span.get("ref_id") in bib_keys:
                    linked_cite_spans += 1
                else:
                    unresolved_cite_spans += 1
    ref_spans = [
        span
        for record in records
        for paragraph in record.get("body_text") or []
        for span in paragraph.get("ref_spans") or []
    ]
    return {
        "total_records": len(records),
        "parse_status_distribution": dict(parse_status.most_common()),
        "has_body_text_count": sum(1 for record in records if record.get("body_text")),
        "has_bib_entries_count": sum(1 for record in records if record.get("bib_entries")),
        "cite_spans_count": len(cite_spans),
        "cite_spans_linked_to_bib_entries_count": linked_cite_spans,
        "unresolved_local_citation_span_count": unresolved_cite_spans,
        "ref_entries_count": sum(len(record.get("ref_entries") or {}) for record in records),
        "ref_spans_count": len(ref_spans),
        "missing_title_count": count_flag(records, "missing_title"),
        "missing_abstract_count": count_flag(records, "missing_abstract"),
        "missing_body_count": count_flag(records, "missing_body"),
        "missing_year_count": count_flag(records, "missing_year"),
        "missing_doi_count": count_flag(records, "missing_doi"),
        "parse_status_distribution": dict(parse_status.most_common()),
        "quality_flags_distribution": dict(flags.most_common()),
        "total_body_text_count": sum(len(record.get("body_text") or []) for record in records),
        "total_bib_entries_count": sum(len(record.get("bib_entries") or {}) for record in records),
    }


def count_flag(records: list[dict], flag: str) -> int:
    return sum(1 for record in records if flag in (record.get("quality_flags") or []))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_report(json_path: Path, md_path: Path, stats: dict) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("# Local S2ORC Quality Report\n\n```json\n" + json.dumps(stats, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")


if __name__ == "__main__":
    main()
