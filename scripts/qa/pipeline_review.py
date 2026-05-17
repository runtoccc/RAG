from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    args = parse_args()
    s2orc = read_json(Path(args.s2orc_report))
    pes2o = read_json(Path(args.pes2o_report))
    passage = read_json(Path(args.passage_report))
    report = build_report(s2orc, pes2o, passage)
    write_report(Path(args.output_json), Path(args.output_md), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review local pipeline alignment.")
    parser.add_argument("--s2orc-report", default="outputs/reports/s2orc_conformance_report.json")
    parser.add_argument("--pes2o-report", default="outputs/reports/pes2o_conformance_report.json")
    parser.add_argument("--passage-report", default="outputs/reports/openscholar_conformance_report.json")
    parser.add_argument("--output-json", default="outputs/reports/pipeline_alignment_review.json")
    parser.add_argument("--output-md", default="outputs/reports/pipeline_alignment_review.md")
    return parser.parse_args()


def build_report(s2orc: dict, pes2o: dict, passage: dict) -> dict:
    strict_enabled = (
        pes2o.get("language_detection_unavailable_count", 0) == 0
        and pes2o.get("unigram_frequency_unavailable_count", 0) == 0
    )
    openscholar_pass = (
        passage.get("all_block_words_lte_256") is True
        and passage.get("label_leak_count", 1) == 0
        and passage.get("not_index_ready_passage_count", 1) == 0
        and passage.get("bad_embedding_text_count", 1) == 0
    )
    chroma_ready = openscholar_pass and passage.get("total_passages", 0) > 0
    return {
        "s2orc_alignment": "local_s2orc_partial",
        "pes2o_alignment": "strict_filter_enabled" if strict_enabled else "strict_filter_disabled",
        "openscholar_alignment": "pass" if openscholar_pass else "fail",
        "chroma_readiness": "pass" if chroma_ready else "fail",
        "top_blockers_before_100k_scale": blockers(s2orc, pes2o, passage),
    }


def blockers(s2orc: dict, pes2o: dict, passage: dict) -> list[str]:
    items = []
    if s2orc.get("cite_spans_count", 0) == 0:
        items.append("S2ORC-compatible object has no TEI citation spans")
    if s2orc.get("unresolved_local_citation_span_count", 0):
        items.append("Some local TEI citation spans do not link to bib_entries")
    if s2orc.get("ref_entries_count", 0) == 0:
        items.append("figure/table/formula ref_entries are none_found")
    if pes2o.get("language_detection_unavailable_count", 0):
        items.append("strict language filtering requires pycld3")
    if pes2o.get("unigram_frequency_unavailable_count", 0):
        items.append("strict unigram probability filtering requires data/resources/unigram_freq.csv")
    if pes2o.get("bad_fallback_title_count", 0) or pes2o.get("title_needs_external_verification_count", 0):
        items.append("metadata enrichment coverage needs manual or external title/abstract verification")
    if sum((pes2o.get("residue_flags_distribution") or {}).values()):
        items.append("non-textual residue leakage remains in some records")
    if not passage.get("all_block_words_lte_256", False):
        items.append("OpenScholar-style passage blocks exceed 256 words")
    items.append("100K scale needs streaming readers, stronger dedup, and versioned/vector collection strategy")
    return items[:5]


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_report(json_path: Path, md_path: Path, report: dict) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("# Pipeline Alignment Review\n\n```json\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")


if __name__ == "__main__":
    main()
