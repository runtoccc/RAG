from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
import math
from pathlib import Path
import re
import unicodedata


try:
    import cld3  # type: ignore
except ImportError:  # pragma: no cover - depends on local optional install
    cld3 = None


VERSION = "local_pes2o_like_v1"
SOURCE = "local_grobid_s2orc_like"
LOCAL_PES2O_DIFFERENCE = "title_abstract_from_grobid_or_filename_not_s2_metadata"
DEFAULT_REPAIRED_INPUT = "data/structured/s2orc_like_repaired.jsonl"
RAW_INPUT_FALLBACK = "data/structured/s2orc_like.jsonl"

SKIP_SECTION_PATTERNS = [
    r"^references?$",
    r"^bibliography$",
    r"^acknowledgements?$",
    r"^funding$",
    r"^author contributions?$",
    r"^conflicts? of interest$",
    r"^competing interests$",
    r"^declaration of interest$",
    r"^disclosure statement$",
    r"^data availability$",
    r"^availability of data$",
    r"^ethics$",
    r"^ethical statement$",
    r"^ethics statement$",
    r"^supplementary material$",
    r"^supplementary information$",
    r"^supporting information$",
    r"^publisher'?s note$",
    r"^author information$",
    r"^correspondence$",
    r"^figure legends?$",
    r"^table legends?$",
    r"^figures?$",
    r"^tables?$",
]

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

END_PUNCTUATION = tuple(".?!:;)]")
ALPHA_WORD_RE = re.compile(r"[A-Za-z]{2,}")
DOI_RE = re.compile(r"\b(?:doi\s*:?\s*)?10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", re.I)
PMID_RE = re.compile(r"\bPMID\s*:?\s*\d+\b", re.I)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
OCR_SPACING_RE = re.compile(r"(?:\b[A-Za-z]\s+){6,}[A-Za-z]\b")
URL_LINE_RE = re.compile(r"^(?:https?://|www\.)\S+$", re.I)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
DATA_AVAILABILITY_RE = re.compile(r"\bdata availability\b|\bavailability of data\b", re.I)
ETHICS_RE = re.compile(r"\bethics?\b|\bethical statement\b", re.I)
SUPPORTING_INFO_RE = re.compile(
    r"\bsupplementary (?:material|information)\b|\bsupporting information\b", re.I
)
ARTICLE_IN_PRESS_RE = re.compile(r"article in press", re.I)
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


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args.input)
    output_all = Path(args.output)
    output_pass = Path(args.pass_output)
    output_failed = Path(args.failed_output)
    unigram_log_probs = load_unigram_log_probs(Path(args.unigram_freq))

    for path in [output_all, output_pass, output_failed]:
        path.parent.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(input_path)
    print(f"[clean] input={input_path}")
    print(f"[clean] records={len(records)}")
    print(f"[clean] output_all={output_all}")
    print(f"[clean] output_pass={output_pass}")
    print(f"[clean] output_failed={output_failed}")

    total = 0
    passed = 0
    failed = 0
    with output_all.open("w", encoding="utf-8") as all_file, output_pass.open(
        "w", encoding="utf-8"
    ) as pass_file, output_failed.open("w", encoding="utf-8") as failed_file:
        for record in records:
            cleaned = clean_record(record, unigram_log_probs)
            line = json.dumps(cleaned, ensure_ascii=False) + "\n"
            all_file.write(line)
            if cleaned["metadata"]["index_ready"]:
                pass_file.write(line)
                passed += 1
            else:
                failed_file.write(line)
                failed += 1
            total += 1
            print(
                f"[ok] {cleaned['paper_id']} index_ready={cleaned['metadata']['index_ready']} "
                f"words={cleaned['metadata']['n_words']} paragraphs={cleaned['metadata']['n_paragraphs']} "
                f"flags={','.join(cleaned['metadata']['quality_flags']) or '-'}"
            )

    print(f"[done] clean_records={total} pass={passed} failed={failed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean S2ORC-like JSONL into peS2o-like JSONL.")
    parser.add_argument("--input", default=DEFAULT_REPAIRED_INPUT)
    parser.add_argument("--output", default="data/clean/pes2o_like.jsonl")
    parser.add_argument("--pass-output", default="data/clean/pes2o_like_pass.jsonl")
    parser.add_argument("--failed-output", default="data/clean/pes2o_like_failed.jsonl")
    parser.add_argument("--unigram-freq", default="data/resources/unigram_freq.csv")
    return parser.parse_args()


def resolve_input_path(input_value: str) -> Path:
    input_path = Path(input_value)
    if input_value == DEFAULT_REPAIRED_INPUT and not input_path.exists():
        print(
            "WARNING: repaired S2ORC-like file not found. Falling back to raw s2orc_like.jsonl."
        )
        return Path(RAW_INPUT_FALLBACK)
    return input_path


def clean_record(record: dict, unigram_log_probs: dict[str, float] | None) -> dict:
    quality_flags = list(record.get("quality_flags") or [])
    metadata = record.get("metadata") or {}
    source_file = record.get("source_file") or metadata.get("source_file") or ""

    raw_title = normalize_text(metadata.get("title") or record.get("title") or "")
    title, title_source, title_flags = choose_title(raw_title, source_file)
    quality_flags.extend(title_flags)
    if not title:
        quality_flags.append("missing_title")

    abstract = normalize_text(record.get("abstract") or "")
    abstract_source = metadata.get("abstract_source") or ("grobid" if abstract else "missing")
    if not abstract:
        quality_flags.append("missing_abstract")

    cleaned_sections, clean_stats = clean_sections(record, unigram_log_probs)
    main_text = "\n\n".join(
        paragraph
        for section in cleaned_sections
        for paragraph in section.get("paragraphs", [])
        if paragraph
    )
    text = "\n\n".join([title, abstract, main_text]).strip()
    words = text.split()
    n_words = len(words)
    n_paragraphs = sum(len(section.get("paragraphs", [])) for section in cleaned_sections)
    n_sections = len(cleaned_sections)
    top_frequencies = top_word_frequencies(words)
    max_alpha_word_ratio = calculate_max_alpha_word_ratio(words)
    language, language_distribution, language_available = detect_language(cleaned_sections)
    year = parse_year(metadata.get("year"))

    if not language_available:
        quality_flags.append("language_detection_unavailable")
    elif language != "en":
        quality_flags.append("non_english")

    if unigram_log_probs is None:
        quality_flags.append("unigram_frequency_unavailable")

    if n_words < 500:
        quality_flags.append("too_short")
    if n_paragraphs < 5:
        quality_flags.append("too_few_paragraphs")
    if max_alpha_word_ratio >= 0.075:
        quality_flags.append("repetitive_text")
    if OCR_SPACING_RE.search(text) or has_severe_spacing_noise(words):
        quality_flags.append("ocr_spacing_noise")
    if year is None:
        quality_flags.append("missing_year")
    elif year <= 1969:
        quality_flags.append("pre_1970")

    residue_flags = detect_quality_audit_residue_flags(main_text)
    quality_flags.extend(residue_flags)

    quality_flags = unique_keep_order(quality_flags)
    pass_hard_filters = passes_hard_filters(
        title=title,
        abstract=abstract,
        n_words=n_words,
        n_paragraphs=n_paragraphs,
        max_alpha_word_ratio=max_alpha_word_ratio,
        year=year,
        language=language,
        language_available=language_available,
        quality_flags=quality_flags,
    )
    pass_quality_audit = passes_quality_audit(quality_flags)
    index_ready = pass_hard_filters and pass_quality_audit
    pass_filters = index_ready

    created_year = year if year and year > 0 else 1970
    paper_id = record.get("paper_id") or stable_id_from_source(source_file)
    return {
        "added": None,
        "created": f"{created_year:04d}-01-01T00:00:00.000Z",
        "id": paper_id,
        "paper_id": paper_id,
        "source": SOURCE,
        "title": title,
        "abstract": abstract,
        "main_text": main_text,
        "text": text,
        "version": VERSION,
        "sections": cleaned_sections,
        "metadata": {
            "source_file": source_file,
            "doi": metadata.get("doi"),
            "year": year,
            "title_source": title_source,
            "abstract_source": abstract_source,
            "local_pes2o_difference_from_official": LOCAL_PES2O_DIFFERENCE,
            "n_words": n_words,
            "n_paragraphs": n_paragraphs,
            "n_sections": n_sections,
            "language": language,
            "language_distribution": language_distribution,
            "max_alpha_word_ratio": max_alpha_word_ratio,
            "top_frequencies": top_frequencies,
            "removed_sections": clean_stats["removed_sections"],
            "removed_section_count": len(clean_stats["removed_sections"]),
            "removed_paragraph_count": clean_stats["removed_paragraph_count"],
            "removed_reference_like_paragraph_count": clean_stats[
                "removed_reference_like_paragraph_count"
            ],
            "removed_noise_paragraph_count": clean_stats["removed_noise_paragraph_count"],
            "removed_low_probability_sections": clean_stats[
                "removed_low_probability_sections"
            ],
            "quality_flags": quality_flags,
            "pass_pes2o_filters": pass_filters,
            "pass_hard_filters": pass_hard_filters,
            "pass_quality_audit": pass_quality_audit,
            "index_ready": index_ready,
            "parse_status": record.get("parse_status"),
        },
    }


def clean_sections(record: dict, unigram_log_probs: dict[str, float] | None) -> tuple[list[dict], dict]:
    removed_sections: list[str] = []
    removed_low_probability_sections: list[str] = []
    removed_reference_like_paragraph_count = 0
    removed_noise_paragraph_count = 0
    cleaned_sections = []
    seen_paragraphs = set()

    for section in record.get("sections") or []:
        section_title = normalize_text(section.get("section_title") or "Unknown")
        if should_skip_section(section_title):
            removed_sections.append(section_title)
            continue

        paragraphs = [normalize_text(paragraph) for paragraph in section.get("paragraphs") or []]
        paragraphs = repair_paragraphs([paragraph for paragraph in paragraphs if paragraph])
        filtered_paragraphs = []
        for paragraph in paragraphs:
            dedup_key = normalize_for_dedup(paragraph)
            if not dedup_key or dedup_key in seen_paragraphs:
                continue
            if looks_like_reference_paragraph(paragraph):
                removed_reference_like_paragraph_count += 1
                continue
            if looks_like_non_textual_paragraph(paragraph):
                removed_noise_paragraph_count += 1
                continue
            seen_paragraphs.add(dedup_key)
            filtered_paragraphs.append(paragraph)

        if not filtered_paragraphs:
            continue
        if unigram_log_probs is not None:
            avg_log_prob = average_log_word_probability(filtered_paragraphs, unigram_log_probs)
            if avg_log_prob < -20:
                removed_low_probability_sections.append(section_title)
                continue

        cleaned_sections.append(
            {"section_title": section_title, "paragraphs": filtered_paragraphs}
        )

    stats = {
        "removed_sections": unique_keep_order(removed_sections),
        "removed_paragraph_count": removed_reference_like_paragraph_count
        + removed_noise_paragraph_count,
        "removed_reference_like_paragraph_count": removed_reference_like_paragraph_count,
        "removed_noise_paragraph_count": removed_noise_paragraph_count,
        "removed_low_probability_sections": unique_keep_order(removed_low_probability_sections),
    }
    return cleaned_sections, stats


def choose_title(raw_title: str, source_file: str) -> tuple[str, str, list[str]]:
    flags = []
    title = raw_title
    source = "grobid"
    bad = is_bad_title(title)
    if len(title) > 220:
        flags.append("very_long_title")
    if bad:
        flags.append("bad_title")
        fallback = title_from_source_file(source_file)
        if fallback:
            title = fallback
            source = "filename_fallback"
            flags.append("title_from_filename_fallback")
            if is_bad_fallback_title(fallback):
                flags.append("bad_fallback_title")
    return title, source, flags


def is_bad_title(title: str) -> bool:
    normalized = normalize_space(title).lower()
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


def is_bad_fallback_title(title: str) -> bool:
    normalized = normalize_space(title).lower()
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


def repair_paragraphs(paragraphs: list[str]) -> list[str]:
    repaired: list[str] = []
    for paragraph in paragraphs:
        if not paragraph:
            continue
        if repaired and should_merge_paragraphs(repaired[-1], paragraph):
            repaired[-1] = normalize_space(f"{repaired[-1]} {paragraph}")
        else:
            repaired.append(paragraph)
    return repaired


def should_merge_paragraphs(previous: str, current: str) -> bool:
    if len(current) < 25 and " " not in current:
        return True
    if has_unclosed_bracket(previous) and has_closing_bracket(current):
        return True
    if previous and not previous.endswith(END_PUNCTUATION):
        return True
    return False


def has_unclosed_bracket(text: str) -> bool:
    return text.count("(") > text.count(")") or text.count("[") > text.count("]")


def has_closing_bracket(text: str) -> bool:
    return ")" in text or "]" in text


def should_skip_section(section_title: str) -> bool:
    normalized = section_title.lower().strip(" .:")
    return any(re.search(pattern, normalized) for pattern in SKIP_SECTION_PATTERNS)


def looks_like_reference_paragraph(text: str) -> bool:
    lower = text.lower()
    words = text.split()
    has_year = bool(YEAR_RE.search(text))
    has_doi_or_pmid = bool(DOI_RE.search(text) or PMID_RE.search(text))
    has_journal_shape = bool(
        re.search(r"\b\d+\s*\(\d+\)\s*:\s*\d+|\b\d+\s*:\s*\d+[-–]\d+", text)
    )
    starts_numbered = bool(re.match(r"^\s*(?:\[\d+\]|\d+\.)\s+[A-Z][A-Za-z-]+", text))
    author_year = bool(
        re.search(r"\b[A-Z][A-Za-z-]+,\s+[A-Z]\.", text)
        and has_year
        and len(words) < 120
    )
    many_authors = lower.count(",") >= 4 and has_year and len(words) < 140
    if starts_numbered and (has_year or has_doi_or_pmid or has_journal_shape):
        return True
    if has_doi_or_pmid and (starts_numbered or author_year or many_authors):
        return True
    if author_year and (has_journal_shape or many_authors):
        return True
    return False


def looks_like_non_textual_paragraph(text: str) -> bool:
    stripped = text.strip()
    lower = stripped.lower()
    words = stripped.split()
    if len(words) < 5 and len(stripped) < 40:
        return True
    if URL_LINE_RE.fullmatch(stripped) or DOI_RE.fullmatch(stripped):
        return True
    if lower.startswith(("fig.", "figure ", "table ")) and len(words) < 80:
        return True
    if "copyright" in lower and len(words) < 120:
        return True
    if "journal homepage" in lower:
        return True
    if "article in press" in lower:
        return True
    if "publisher" in lower and "note" in lower and len(words) < 120:
        return True
    if ("license" in lower or "creative commons" in lower) and len(words) < 160:
        return True
    if "correspondence" in lower and EMAIL_RE.search(stripped) and len(words) < 80:
        return True
    if EMAIL_RE.fullmatch(stripped):
        return True
    if looks_like_table_fragment(stripped):
        return True
    return False


def looks_like_table_fragment(text: str) -> bool:
    tokens = text.split()
    if len(tokens) < 12:
        return False
    numeric_tokens = sum(1 for token in tokens if re.search(r"\d", token))
    symbol_tokens = sum(1 for token in tokens if re.fullmatch(r"[-+<>=.,;:/()0-9]+", token))
    short_tokens = sum(1 for token in tokens if len(token) <= 2)
    return (numeric_tokens + symbol_tokens + short_tokens * 0.25) / len(tokens) > 0.55


def detect_language(sections: list[dict]) -> tuple[str, dict[str, int], bool]:
    if cld3 is None:
        return "unknown", {}, False
    counts: Counter[str] = Counter()
    for section in sections:
        for paragraph in section.get("paragraphs", []):
            result = cld3.get_language(paragraph[:2000])
            if result and result.language:
                counts[result.language] += 1
    if not counts:
        return "unknown", {}, True
    language = counts.most_common(1)[0][0]
    return language, dict(counts.most_common()), True


def load_unigram_log_probs(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    rows = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            word = normalize_space(row.get("word") or row.get("token") or "").lower()
            raw_value = row.get("freq") or row.get("frequency") or row.get("count") or row.get("prob")
            if not word or raw_value is None:
                continue
            try:
                value = float(raw_value)
            except ValueError:
                continue
            if value > 0:
                rows.append((word, value))
    if not rows:
        return None
    total = sum(value for _, value in rows)
    return {word: math.log(value / total) for word, value in rows}


def average_log_word_probability(paragraphs: list[str], log_probs: dict[str, float]) -> float:
    words = [
        token.lower()
        for paragraph in paragraphs
        for token in ALPHA_WORD_RE.findall(paragraph)
    ]
    if not words:
        return -100.0
    values = [log_probs.get(word, -30.0) for word in words]
    return sum(values) / len(values)


def detect_quality_audit_residue_flags(text: str) -> list[str]:
    flags = []
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    if any(looks_like_reference_paragraph(paragraph) for paragraph in paragraphs):
        flags.append("reference_residue")
    if DATA_AVAILABILITY_RE.search(text):
        flags.append("data_availability_residue")
    if ETHICS_RE.search(text):
        flags.append("ethics_residue")
    if SUPPORTING_INFO_RE.search(text):
        flags.append("supporting_information_residue")
    if ARTICLE_IN_PRESS_RE.search(text):
        flags.append("article_in_press_residue")
    return flags


def passes_hard_filters(
    title: str,
    abstract: str,
    n_words: int,
    n_paragraphs: int,
    max_alpha_word_ratio: float,
    year: int | None,
    language: str,
    language_available: bool,
    quality_flags: list[str],
) -> bool:
    if not title or not abstract:
        return False
    if n_words < 500 or n_paragraphs < 5:
        return False
    if max_alpha_word_ratio >= 0.075:
        return False
    if year is not None and year <= 1969:
        return False
    if "ocr_spacing_noise" in quality_flags:
        return False
    if language_available and language != "en":
        return False
    return True


def passes_quality_audit(quality_flags: list[str]) -> bool:
    blocking_flags = {
        "bad_title",
        "bad_fallback_title",
        "reference_residue",
        "data_availability_residue",
        "ethics_residue",
        "supporting_information_residue",
        "article_in_press_residue",
    }
    return not bool(blocking_flags & set(quality_flags))


def top_word_frequencies(words: list[str], limit: int = 20) -> list[dict]:
    normalized = [normalize_token(word) for word in words]
    normalized = [word for word in normalized if word]
    counter = Counter(normalized)
    total = len(normalized) or 1
    return [
        {"word": word, "count": count, "ratio": count / total}
        for word, count in counter.most_common(limit)
    ]


def calculate_max_alpha_word_ratio(words: list[str]) -> float:
    alpha_words = [normalize_token(word) for word in words if ALPHA_WORD_RE.search(word)]
    alpha_words = [word for word in alpha_words if word]
    if not alpha_words:
        return 0.0
    return Counter(alpha_words).most_common(1)[0][1] / len(alpha_words)


def has_severe_spacing_noise(words: list[str]) -> bool:
    if len(words) < 80:
        return False
    one_char = sum(1 for word in words if len(normalize_token(word)) == 1)
    return one_char / len(words) > 0.18


def parse_year(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    match = YEAR_RE.search(str(value))
    return int(match.group(0)) if match else None


def stable_id_from_source(source_file: str) -> str:
    stem = Path(source_file).stem or "unknown"
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower() or "unknown"


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


def normalize_text(text: str | None) -> str:
    return normalize_space(unicodedata.normalize("NFC", text or ""))


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_token(token: str) -> str:
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", token.lower())


def normalize_for_dedup(text: str) -> str:
    return re.sub(r"\W+", "", text.lower())


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
