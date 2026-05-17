from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


RESIDUE_FLAGS = [
    "reference_residue",
    "data_availability_residue",
    "ethics_residue",
    "supporting_information_residue",
    "article_in_press_residue",
]


def main() -> None:
    args = parse_args()
    records = read_jsonl(Path(args.input))
    stats = build_stats(records)
    write_report(Path(args.report_json), Path(args.report_md), stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local peS2o-style clean/filter quality.")
    parser.add_argument("--input", default="data/clean/pes2o_style.jsonl")
    parser.add_argument("--report-json", default="outputs/reports/pes2o_conformance_report.json")
    parser.add_argument("--report-md", default="outputs/reports/pes2o_conformance_report.md")
    return parser.parse_args()


def build_stats(records: list[dict]) -> dict:
    flags = Counter(flag for record in records for flag in quality_flags(record))
    fail_reasons = Counter(reason for record in records for reason in metadata(record).get("fail_reasons") or [])
    hard_fail_reasons = Counter(
        reason for record in records for reason in metadata(record).get("hard_fail_reasons") or []
    )
    soft_warnings = Counter(
        warning for record in records for warning in metadata(record).get("soft_warnings") or []
    )
    removed_sections = Counter(section for record in records for section in metadata(record).get("removed_sections") or [])
    residue_flags = Counter(flag for flag, count in flags.items() if flag in RESIDUE_FLAGS for _ in range(count))
    pass_count = sum(1 for record in records if metadata(record).get("index_ready"))
    return {
        "total": len(records),
        "pass": pass_count,
        "failed_count": len(records) - pass_count,
        "fail_reasons_distribution": dict(fail_reasons.most_common()),
        "hard_fail_reasons_distribution": dict(hard_fail_reasons.most_common()),
        "soft_warnings_distribution": dict(soft_warnings.most_common()),
        "title_language_or_logprob_fail_count": decision_fail_count(
            records, "title_language_or_logprob_ok"
        ),
        "abstract_language_or_logprob_fail_count": decision_fail_count(
            records, "abstract_language_or_logprob_ok"
        ),
        "document_language_fail_count": decision_fail_count(records, "document_language_ok"),
        "unigram_filter_active": flags.get("unigram_frequency_unavailable", 0) == 0,
        "cld3_active": flags.get("language_detection_unavailable", 0) == 0,
        "max_alpha_word_ratio_fail_count": decision_fail_count(
            records, "max_alpha_word_ratio_ok"
        ),
        "n_words_min": min((metadata(record).get("n_words") or 0 for record in records), default=0),
        "n_paragraphs_min": min(
            (metadata(record).get("n_paragraphs") or 0 for record in records), default=0
        ),
        "language_detection_unavailable_count": flags.get("language_detection_unavailable", 0),
        "unigram_frequency_unavailable_count": flags.get("unigram_frequency_unavailable", 0),
        "missing_year_count": flags.get("missing_year", 0),
        "bad_fallback_title_count": flags.get("bad_fallback_title", 0),
        "title_needs_external_verification_count": flags.get("title_needs_external_verification", 0),
        "removed_sections_distribution": dict(removed_sections.most_common()),
        "residue_flags_distribution": dict(residue_flags.most_common()),
        "quality_flags_distribution": dict(flags.most_common()),
    }


def metadata(record: dict) -> dict:
    return record.get("metadata") or {}


def quality_flags(record: dict) -> list[str]:
    return metadata(record).get("quality_flags") or []


def decision_fail_count(records: list[dict], decision: str) -> int:
    return sum(
        1
        for record in records
        if (metadata(record).get("filter_decisions") or {}).get(decision) is False
    )


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_report(json_path: Path, md_path: Path, stats: dict) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("# peS2o-style Quality Report\n\n```json\n" + json.dumps(stats, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")


if __name__ == "__main__":
    main()
