from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re


REFERENCE_RE = re.compile(
    r"(?:^\s*(?:\[\d+\]|\d+\.)\s+[A-Z][A-Za-z-]+.*\b(?:19|20)\d{2}\b|"
    r"\bdoi\s*:?\s*10\.|"
    r"\bPMID\s*:?\s*\d+\b)",
    re.I,
)
ARTICLE_IN_PRESS_RE = re.compile(r"article in press", re.I)
FIGURE_TABLE_RE = re.compile(r"^\s*(?:fig\.?|figure|table)\s+\d+", re.I | re.M)
DATA_AVAILABILITY_RE = re.compile(r"\bdata availability\b|\bavailability of data\b", re.I)
ETHICS_RE = re.compile(r"\bethics?\b|\bethical statement\b", re.I)
SUPPORTING_INFO_RE = re.compile(
    r"\bsupplementary (?:material|information)\b|\bsupporting information\b", re.I
)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    report_md = Path(args.report_md)
    report_json = Path(args.report_json)
    suspicious_jsonl = Path(args.suspicious_jsonl)
    for path in [report_md, report_json, suspicious_jsonl]:
        path.parent.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(input_path)
    stats, suspicious = build_stats(records)
    report_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(build_markdown(stats), encoding="utf-8")
    with suspicious_jsonl.open("w", encoding="utf-8") as file:
        for item in suspicious:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[pes2o-quality] input={input_path}")
    print(f"[pes2o-quality] total_records={stats['total_records']}")
    print(
        f"[pes2o-quality] pass={stats['pass_count']} failed={stats['failed_count']} "
        f"pass_ratio={stats['pass_ratio']:.3f}"
    )
    print(f"[pes2o-quality] reference_residue_count={stats['reference_residue_count']}")
    print(f"[pes2o-quality] article_in_press_residue_count={stats['article_in_press_residue_count']}")
    print(f"[pes2o-quality] report_md={report_md}")
    print(f"[pes2o-quality] report_json={report_json}")
    print(f"[pes2o-quality] suspicious_jsonl={suspicious_jsonl}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local peS2o-like clean JSONL quality.")
    parser.add_argument("--input", default="data/clean/pes2o_like.jsonl")
    parser.add_argument("--report-md", default="outputs/reports/pes2o_clean_quality_report.md")
    parser.add_argument("--report-json", default="outputs/reports/pes2o_clean_quality_report.json")
    parser.add_argument(
        "--suspicious-jsonl",
        default="outputs/reports/pes2o_clean_suspicious_examples.jsonl",
    )
    return parser.parse_args()


def build_stats(records: list[dict]) -> tuple[dict, list[dict]]:
    word_counts = [metadata(record).get("n_words") or 0 for record in records]
    paragraph_counts = [metadata(record).get("n_paragraphs") or 0 for record in records]
    pass_count = sum(1 for record in records if metadata(record).get("index_ready"))
    pass_hard_filters_count = sum(
        1 for record in records if metadata(record).get("pass_hard_filters")
    )
    pass_quality_audit_count = sum(
        1 for record in records if metadata(record).get("pass_quality_audit")
    )
    flag_counter = Counter(flag for record in records for flag in quality_flags(record))
    removed_sections_counter = Counter(
        section for record in records for section in metadata(record).get("removed_sections") or []
    )
    removed_low_prob_counter = Counter(
        section
        for record in records
        for section in metadata(record).get("removed_low_probability_sections") or []
    )

    reference_residue_records = residue_records(records, REFERENCE_RE)
    article_in_press_records = residue_records(records, ARTICLE_IN_PRESS_RE)
    figure_table_records = residue_records(records, FIGURE_TABLE_RE)
    data_availability_records = residue_records(records, DATA_AVAILABILITY_RE)
    ethics_records = residue_records(records, ETHICS_RE)
    supporting_info_records = residue_records(records, SUPPORTING_INFO_RE)

    stats = {
        "total_records": len(records),
        "pass_count": pass_count,
        "index_ready_count": pass_count,
        "failed_count": len(records) - pass_count,
        "pass_ratio": pass_count / len(records) if records else 0.0,
        "pass_hard_filters_count": pass_hard_filters_count,
        "pass_quality_audit_count": pass_quality_audit_count,
        "avg_words": average(word_counts),
        "min_words": min(word_counts) if word_counts else 0,
        "max_words": max(word_counts) if word_counts else 0,
        "avg_paragraphs": average(paragraph_counts),
        "min_paragraphs": min(paragraph_counts) if paragraph_counts else 0,
        "max_paragraphs": max(paragraph_counts) if paragraph_counts else 0,
        "missing_title_count": count_flag(records, "missing_title"),
        "missing_abstract_count": count_flag(records, "missing_abstract"),
        "bad_title_count": count_flag(records, "bad_title"),
        "bad_fallback_title_count": count_flag(records, "bad_fallback_title"),
        "very_long_title_count": count_flag(records, "very_long_title"),
        "title_from_filename_fallback_count": count_flag(records, "title_from_filename_fallback"),
        "metadata_override_applied_count": count_flag(records, "metadata_override_applied"),
        "abstract_from_section_recovery_count": count_flag(records, "abstract_from_section_recovery"),
        "still_missing_abstract_count": count_flag(records, "missing_abstract"),
        "main_text_missing_count": sum(1 for record in records if not record.get("main_text")),
        "non_english_count": count_flag(records, "non_english"),
        "language_detection_unavailable_count": count_flag(records, "language_detection_unavailable"),
        "unigram_frequency_unavailable_count": count_flag(records, "unigram_frequency_unavailable"),
        "too_short_count": count_flag(records, "too_short"),
        "too_few_paragraphs_count": count_flag(records, "too_few_paragraphs"),
        "repetitive_text_count": count_flag(records, "repetitive_text"),
        "ocr_spacing_noise_count": count_flag(records, "ocr_spacing_noise"),
        "reference_residue_count": len(reference_residue_records),
        "article_in_press_residue_count": len(article_in_press_records),
        "figure_table_residue_count": len(figure_table_records),
        "data_availability_residue_count": len(data_availability_records),
        "ethics_residue_count": len(ethics_records),
        "supporting_information_residue_count": len(supporting_info_records),
        "quality_flags_distribution": dict(flag_counter.most_common()),
        "failed_reasons_top_distribution": dict(failed_reasons(records).most_common()),
        "removed_sections_distribution": dict(removed_sections_counter.most_common()),
        "removed_low_probability_sections_distribution": dict(removed_low_prob_counter.most_common()),
    }

    suspicious = []
    suspicious.extend(label_examples("bad_title", records_with_flag(records, "bad_title")))
    suspicious.extend(label_examples("reference_residue", reference_residue_records))
    suspicious.extend(label_examples("article_in_press_residue", article_in_press_records))
    suspicious.extend(label_examples("failed", [r for r in records if not metadata(r).get("index_ready")]))
    suspicious.extend(label_examples("pass", [r for r in records if metadata(r).get("index_ready")]))
    return stats, suspicious


def build_markdown(stats: dict) -> str:
    lines = [
        "# peS2o-like Clean Quality Report",
        "",
        f"- total_records: {stats['total_records']}",
        f"- index_ready_count: {stats['index_ready_count']}",
        f"- failed_count: {stats['failed_count']}",
        f"- pass_ratio: {stats['pass_ratio']:.3f}",
        f"- pass_hard_filters_count: {stats['pass_hard_filters_count']}",
        f"- pass_quality_audit_count: {stats['pass_quality_audit_count']}",
        f"- avg_words: {stats['avg_words']:.2f}",
        f"- min_words: {stats['min_words']}",
        f"- max_words: {stats['max_words']}",
        f"- avg_paragraphs: {stats['avg_paragraphs']:.2f}",
        f"- min_paragraphs: {stats['min_paragraphs']}",
        f"- max_paragraphs: {stats['max_paragraphs']}",
        f"- missing_title_count: {stats['missing_title_count']}",
        f"- missing_abstract_count: {stats['missing_abstract_count']}",
        f"- bad_title_count: {stats['bad_title_count']}",
        f"- bad_fallback_title_count: {stats['bad_fallback_title_count']}",
        f"- metadata_override_applied_count: {stats['metadata_override_applied_count']}",
        f"- abstract_from_section_recovery_count: {stats['abstract_from_section_recovery_count']}",
        f"- still_missing_abstract_count: {stats['still_missing_abstract_count']}",
        f"- main_text_missing_count: {stats['main_text_missing_count']}",
        f"- reference_residue_count: {stats['reference_residue_count']}",
        f"- article_in_press_residue_count: {stats['article_in_press_residue_count']}",
        "",
        "## Quality Flags Distribution",
        json.dumps(stats["quality_flags_distribution"], ensure_ascii=False, indent=2),
        "",
        "## Failed Reasons Top Distribution",
        json.dumps(stats["failed_reasons_top_distribution"], ensure_ascii=False, indent=2),
        "",
        "## Removed Sections Distribution",
        json.dumps(stats["removed_sections_distribution"], ensure_ascii=False, indent=2),
        "",
        "## Removed Low Probability Sections Distribution",
        json.dumps(
            stats["removed_low_probability_sections_distribution"],
            ensure_ascii=False,
            indent=2,
        ),
    ]
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


def residue_records(records: list[dict], pattern: re.Pattern, limit: int | None = None) -> list[dict]:
    selected = []
    for record in records:
        if pattern.search(record.get("text") or ""):
            selected.append(record)
            if limit and len(selected) >= limit:
                break
    return selected


def records_with_flag(records: list[dict], flag: str) -> list[dict]:
    return [record for record in records if flag in quality_flags(record)]


def label_examples(label: str, records: list[dict], limit: int = 5) -> list[dict]:
    examples = []
    for record in records[:limit]:
        examples.append(
            {
                "label": label,
                "paper_id": record.get("paper_id"),
                "source_file": metadata(record).get("source_file"),
                "title": record.get("title"),
                "quality_flags": quality_flags(record),
                "text_preview": preview(record.get("text") or "", 500),
            }
        )
    return examples


def metadata(record: dict) -> dict:
    return record.get("metadata") or {}


def quality_flags(record: dict) -> list[str]:
    return metadata(record).get("quality_flags") or []


def count_flag(records: list[dict], flag: str) -> int:
    return sum(1 for record in records if flag in quality_flags(record))


def failed_reasons(records: list[dict]) -> Counter:
    counter: Counter[str] = Counter()
    for record in records:
        if metadata(record).get("index_ready"):
            continue
        flags = quality_flags(record)
        if flags:
            counter.update(flags)
        elif not metadata(record).get("pass_hard_filters"):
            counter["failed_hard_filters"] += 1
        elif not metadata(record).get("pass_quality_audit"):
            counter["failed_quality_audit"] += 1
        else:
            counter["unknown"] += 1
    return counter


def average(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def preview(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


if __name__ == "__main__":
    main()
