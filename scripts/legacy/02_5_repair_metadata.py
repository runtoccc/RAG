from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import re
import unicodedata


BAD_TITLE_EXACT = {
    "article",
    "research article",
    "original article",
    "review",
    "short communication",
}
BAD_TITLE_PHRASES = [
    "article in press",
    "nature ecology & evolution article",
    "comparative biochemistry and physiology, part b",
]
JOURNAL_TITLE_HINTS = [
    "journal of",
    "comparative biochemistry",
    "aquaculture",
    "fisheries science",
    "marine biology",
]
BAD_FALLBACK_PATTERNS = [
    r"\bs2\.0\b",
    r"\bmain\b",
    r"\bannurev\b",
    r"\barticle\b",
    r"\bresearch article\b",
    r"\boriginal article\b",
    r"\bpdf\b",
]
DOI_ONLY_RE = re.compile(r"^(?:doi[_\s-]*)?10[._]\d{4,9}[/._-].+", re.I)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    override_path = Path(args.override)
    output_path = Path(args.output)
    report_md = Path(args.report_md)
    report_json = Path(args.report_json)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(input_path)
    overrides = read_overrides(override_path)
    repaired_records = []
    before = summarize_quality(records)
    counters: Counter[str] = Counter()
    repaired_examples = []
    unrepaired_examples = []

    for record in records:
        repaired, events = repair_record(record, overrides)
        repaired_records.append(repaired)
        for event in events:
            counters[event] += 1
        if events and len(repaired_examples) < 10:
            repaired_examples.append(example(repaired, events))
        if has_unrepaired_metadata(repaired) and len(unrepaired_examples) < 10:
            unrepaired_examples.append(example(repaired, ["unrepaired_metadata"]))

    after = summarize_quality(repaired_records)
    write_jsonl(output_path, repaired_records)
    report = {
        "total_records": len(records),
        "missing_abstract_before": before["missing_abstract"],
        "missing_abstract_after": after["missing_abstract"],
        "bad_title_before": before["bad_title"],
        "bad_title_after": after["bad_title"],
        "override_applied_count": counters["metadata_override_csv"],
        "section_abstract_recovery_count": counters["section_abstract_recovery"],
        "filename_fallback_count": counters["title_from_filename_fallback"],
        "bad_fallback_title_count": after["bad_fallback_title"],
        "still_missing_abstract_count": after["missing_abstract"],
        "repaired_examples": repaired_examples,
        "unrepaired_examples": unrepaired_examples,
        "recommended_next_action": recommended_next_action(after),
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(build_markdown(report), encoding="utf-8")

    print(f"[metadata-repair] input={input_path}")
    print(f"[metadata-repair] override={override_path}")
    print(f"[metadata-repair] output={output_path}")
    print(
        "[metadata-repair] missing_abstract "
        f"{report['missing_abstract_before']} -> {report['missing_abstract_after']}"
    )
    print(f"[metadata-repair] override_applied_count={report['override_applied_count']}")
    print(
        "[metadata-repair] section_abstract_recovery_count="
        f"{report['section_abstract_recovery_count']}"
    )
    print(f"[metadata-repair] bad_fallback_title_count={report['bad_fallback_title_count']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair local S2ORC-like metadata before peS2o-like filtering.")
    parser.add_argument("--input", default="data/structured/s2orc_like.jsonl")
    parser.add_argument("--override", default="data/metadata/metadata_override.csv")
    parser.add_argument("--output", default="data/structured/s2orc_like_repaired.jsonl")
    parser.add_argument("--report-md", default="outputs/reports/metadata_repair_report.md")
    parser.add_argument("--report-json", default="outputs/reports/metadata_repair_report.json")
    return parser.parse_args()


def repair_record(record: dict, overrides: dict[str, dict]) -> tuple[dict, list[str]]:
    record = json.loads(json.dumps(record, ensure_ascii=False))
    metadata = record.setdefault("metadata", {})
    events: list[str] = []
    notes: list[str] = []
    source_file = record.get("source_file") or metadata.get("source_file") or ""
    doi = normalize_doi(metadata.get("doi"))
    override = find_override(source_file, doi, overrides)

    if override:
        events.append("metadata_override_csv")
        notes.append(override.get("notes") or "")
        apply_override(record, override)
        metadata = record.setdefault("metadata", {})
        source_file = record.get("source_file") or metadata.get("source_file") or source_file

    if not normalize_text(record.get("abstract")):
        recovered = recover_abstract_from_sections(record)
        if recovered:
            record["abstract"] = recovered
            record["abstract_paragraphs"] = [
                {
                    "text": recovered,
                    "section": "Abstract",
                    "cite_spans": [],
                    "ref_spans": [],
                }
            ]
            metadata["abstract_source"] = "section_abstract_recovery"
            events.append("section_abstract_recovery")
        else:
            metadata["abstract_source"] = metadata.get("abstract_source") or "missing"
    else:
        metadata["abstract_source"] = metadata.get("abstract_source") or (
            "metadata_override_csv" if override and override.get("abstract") else "grobid"
        )

    title = normalize_text(record.get("title") or metadata.get("title"))
    if is_bad_title(title):
        if override and override.get("title"):
            title = normalize_text(override["title"])
        else:
            fallback = title_from_source_file(source_file)
            if fallback:
                title = fallback
                metadata["title_source"] = "filename_fallback"
                events.append("title_from_filename_fallback")
    else:
        metadata["title_source"] = metadata.get("title_source") or (
            "metadata_override_csv" if override and override.get("title") else "grobid"
        )

    record["title"] = title
    metadata["title"] = title
    if override and override.get("doi"):
        metadata["doi"] = normalize_doi(override.get("doi"))
    if override and override.get("year"):
        metadata["year"] = parse_year(override.get("year"))

    flags = recompute_quality_flags(
        record,
        title_source=metadata.get("title_source"),
        metadata_override_applied="metadata_override_csv" in events,
    )
    record["quality_flags"] = flags
    record["metadata_repair"] = {
        "applied": bool(events),
        "sources": unique_keep_order(events),
        "notes": [note for note in notes if note],
    }
    return record, unique_keep_order(events)


def apply_override(record: dict, override: dict) -> None:
    metadata = record.setdefault("metadata", {})
    if override.get("title"):
        title = normalize_text(override["title"])
        record["title"] = title
        metadata["title"] = title
        metadata["title_source"] = "metadata_override_csv"
    if override.get("abstract"):
        abstract = normalize_text(override["abstract"])
        record["abstract"] = abstract
        record["abstract_paragraphs"] = [
            {
                "text": abstract,
                "section": "Abstract",
                "cite_spans": [],
                "ref_spans": [],
            }
        ]
        metadata["abstract_source"] = "metadata_override_csv"
    if override.get("year"):
        metadata["year"] = parse_year(override["year"])
    if override.get("doi"):
        metadata["doi"] = normalize_doi(override["doi"])


def recover_abstract_from_sections(record: dict) -> str:
    wanted = ("abstract", "summary", "author summary")
    for section in record.get("sections") or []:
        title = normalize_text(section.get("section_title")).lower()
        if any(name in title for name in wanted):
            paragraphs = [normalize_text(p) for p in section.get("paragraphs") or []]
            paragraphs = [p for p in paragraphs if p]
            if paragraphs:
                return "\n\n".join(paragraphs)
    return ""


def recompute_quality_flags(
    record: dict,
    title_source: str | None,
    metadata_override_applied: bool,
) -> list[str]:
    metadata = record.get("metadata") or {}
    title = normalize_text(record.get("title") or metadata.get("title"))
    abstract = normalize_text(record.get("abstract"))
    doi = normalize_doi(metadata.get("doi"))
    year = parse_year(metadata.get("year"))
    flags = []
    if not title:
        flags.append("missing_title")
    if not abstract:
        flags.append("missing_abstract")
    if year is None:
        flags.append("missing_year")
    if not doi:
        flags.append("missing_doi")
    if is_bad_title(title):
        flags.append("bad_title")
    if title_source == "filename_fallback":
        flags.append("title_from_filename_fallback")
        if is_bad_fallback_title(title):
            flags.append("bad_fallback_title")
    if (metadata.get("abstract_source") or "") == "section_abstract_recovery":
        flags.append("abstract_from_section_recovery")
    if metadata_override_applied:
        flags.append("metadata_override_applied")
    return unique_keep_order(flags)


def summarize_quality(records: list[dict]) -> dict[str, int]:
    return {
        "missing_abstract": sum(1 for record in records if not normalize_text(record.get("abstract"))),
        "bad_title": sum(1 for record in records if is_bad_title(record.get("title") or "")),
        "bad_fallback_title": sum(
            1 for record in records if "bad_fallback_title" in (record.get("quality_flags") or [])
        ),
    }


def is_bad_title(title: str | None) -> bool:
    normalized = normalize_text(title).lower()
    if not normalized or len(normalized) < 15:
        return True
    if normalized in BAD_TITLE_EXACT:
        return True
    if any(phrase in normalized for phrase in BAD_TITLE_PHRASES):
        return True
    words = normalized.split()
    if len(words) <= 6 and any(hint in normalized for hint in JOURNAL_TITLE_HINTS):
        return True
    if len(words) <= 5 and any(word in normalized for word in ["article", "review"]):
        return True
    return False


def is_bad_fallback_title(title: str | None) -> bool:
    normalized = normalize_text(title).lower()
    if len(normalized) < 15:
        return True
    if DOI_ONLY_RE.match(normalized):
        return True
    if is_bad_title(normalized):
        return True
    return any(re.search(pattern, normalized) for pattern in BAD_FALLBACK_PATTERNS)


def title_from_source_file(source_file: str) -> str:
    stem = Path(source_file).stem
    stem = re.sub(r"^\d+[\s._-]+", "", stem)
    stem = re.sub(r"[_-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return normalize_text(stem)


def has_unrepaired_metadata(record: dict) -> bool:
    flags = record.get("quality_flags") or []
    return bool({"missing_abstract", "missing_title", "bad_title", "bad_fallback_title"} & set(flags))


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


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_overrides(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    overrides: dict[str, dict] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            normalized = {key: normalize_text(value) for key, value in row.items()}
            source_file = normalized.get("source_file") or ""
            doi = normalize_doi(normalized.get("doi"))
            if source_file:
                overrides[f"source:{source_file.lower()}"] = normalized
            if doi:
                overrides[f"doi:{doi}"] = normalized
    return overrides


def find_override(source_file: str, doi: str | None, overrides: dict[str, dict]) -> dict | None:
    source_key = f"source:{source_file.lower()}"
    if source_key in overrides:
        return overrides[source_key]
    if doi and f"doi:{doi}" in overrides:
        return overrides[f"doi:{doi}"]
    return None


def recommended_next_action(after: dict[str, int]) -> str:
    if after["missing_abstract"]:
        return "Add manual title/abstract rows to data/metadata/metadata_override.csv for remaining missing abstracts."
    if after["bad_fallback_title"]:
        return "Review bad fallback titles and add metadata overrides before indexing."
    return "Proceed to peS2o-like cleaning and OpenScholar-style passage construction."


def build_markdown(report: dict) -> str:
    lines = [
        "# Metadata Repair Report",
        "",
        f"- total_records: {report['total_records']}",
        f"- missing_abstract_before: {report['missing_abstract_before']}",
        f"- missing_abstract_after: {report['missing_abstract_after']}",
        f"- bad_title_before: {report['bad_title_before']}",
        f"- bad_title_after: {report['bad_title_after']}",
        f"- override_applied_count: {report['override_applied_count']}",
        f"- section_abstract_recovery_count: {report['section_abstract_recovery_count']}",
        f"- filename_fallback_count: {report['filename_fallback_count']}",
        f"- bad_fallback_title_count: {report['bad_fallback_title_count']}",
        f"- still_missing_abstract_count: {report['still_missing_abstract_count']}",
        "",
        "## Recommended Next Action",
        report["recommended_next_action"],
        "",
        "## Repaired Examples",
        json.dumps(report["repaired_examples"], ensure_ascii=False, indent=2),
        "",
        "## Unrepaired Examples",
        json.dumps(report["unrepaired_examples"], ensure_ascii=False, indent=2),
    ]
    return "\n".join(lines) + "\n"


def example(record: dict, events: list[str]) -> dict:
    return {
        "paper_id": record.get("paper_id"),
        "source_file": record.get("source_file"),
        "title": record.get("title"),
        "quality_flags": record.get("quality_flags") or [],
        "events": events,
    }


def normalize_text(text: object) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text or ""))).strip()


def normalize_doi(value: object) -> str | None:
    text = normalize_text(value).lower()
    if not text:
        return None
    text = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", text, flags=re.I)
    return text.rstrip(".,;") or None


def parse_year(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    match = YEAR_RE.search(str(value))
    return int(match.group(0)) if match else None


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
