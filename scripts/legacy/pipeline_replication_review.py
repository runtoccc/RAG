from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    args = parse_args()
    s2orc = read_json(Path(args.s2orc_report))
    repair = read_json(Path(args.metadata_repair_report))
    pes2o = read_json(Path(args.pes2o_report))
    passage = read_json(Path(args.passage_report))
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    report = build_report(s2orc, repair, pes2o, passage)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(build_markdown(report), encoding="utf-8")

    print(f"[pipeline-review] output_json={output_json}")
    print(f"[pipeline-review] output_md={output_md}")
    print(f"[pipeline-review] s2orc_status={report['s2orc_replication_status']['conclusion']}")
    print(f"[pipeline-review] pes2o_status={report['pes2o_replication_status']['conclusion']}")
    print(f"[pipeline-review] openscholar_status={report['openscholar_passage_status']['conclusion']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review local replication alignment against S2ORC, peS2o, and OpenScholar.")
    parser.add_argument("--s2orc-report", default="outputs/reports/s2orc_like_quality_report.json")
    parser.add_argument("--metadata-repair-report", default="outputs/reports/metadata_repair_report.json")
    parser.add_argument("--pes2o-report", default="outputs/reports/pes2o_clean_quality_report.json")
    parser.add_argument("--passage-report", default="outputs/reports/passage_quality_report.json")
    parser.add_argument("--output-json", default="outputs/reports/pipeline_replication_review.json")
    parser.add_argument("--output-md", default="outputs/reports/pipeline_replication_review.md")
    return parser.parse_args()


def build_report(s2orc: dict, repair: dict, pes2o: dict, passage: dict) -> dict:
    passage_stats = passage.get("stats") or {}
    s2orc_status = {
        "structured_json_present": bool(s2orc.get("total_records")),
        "body_text_present": bool(s2orc.get("total_body_text_count")),
        "bib_entries_present": bool(s2orc.get("total_bib_entries_count")),
        "cite_spans_implemented": False,
        "ref_entries_implemented": False,
        "canonical_metadata_implemented": False,
        "conclusion": "partial" if s2orc.get("total_body_text_count") else "skeleton",
    }
    pes2o_status = {
        "title_abstract_required": True,
        "all_pass_failed_split": True,
        "english_filtering_available": pes2o.get("language_detection_unavailable_count", 0) == 0,
        "unigram_probability_available": pes2o.get("unigram_frequency_unavailable_count", 0) == 0,
        "500_words_filter": True,
        "5_paragraphs_filter": True,
        "alpha_word_ratio_filter": True,
        "non_textual_residue_count": sum(
            int(pes2o.get(key, 0))
            for key in [
                "reference_residue_count",
                "data_availability_residue_count",
                "ethics_residue_count",
                "supporting_information_residue_count",
                "article_in_press_residue_count",
            ]
        ),
        "index_ready_count": pes2o.get("index_ready_count", pes2o.get("pass_count", 0)),
        "failed_count": pes2o.get("failed_count", 0),
        "conclusion": pes2o_conclusion(pes2o),
    }
    openscholar_status = {
        "256_word_blocks": passage_stats.get("ratio_blocks_eq_256", 0) >= 0.8,
        "title_prefix": passage_stats.get("title_prefix_invalid_count", 1) == 0,
        "section_labels_absent": passage_stats.get("section_label_leak_count", 1) == 0,
        "input_only_pass_index_ready_records": passage_stats.get("not_index_ready_passage_count", 1) == 0,
        "conclusion": "pass"
        if passage_stats.get("title_prefix_invalid_count", 1) == 0
        and passage_stats.get("section_label_leak_count", 1) == 0
        and passage_stats.get("not_index_ready_passage_count", 1) == 0
        else "needs fix",
    }
    chroma_status = {
        "embedding_text_used": True,
        "batch_add_implemented": True,
        "large_scale_risk_notes": [
            "populate_database.load_documents still materializes all passages as a list.",
            "Chroma db.get(include=[]) can become slow on large collections.",
            "Use versioned collections or batch existence checks before 100K-paper scale.",
        ],
    }
    blockers = build_blockers(repair, pes2o, passage_stats)
    return {
        "s2orc_replication_status": s2orc_status,
        "pes2o_replication_status": pes2o_status,
        "openscholar_passage_status": openscholar_status,
        "chroma_indexing_status": chroma_status,
        "top_blockers_before_100k_scale": blockers,
    }


def pes2o_conclusion(pes2o: dict) -> str:
    if pes2o.get("index_ready_count", pes2o.get("pass_count", 0)) == 0:
        return "blocked"
    if pes2o.get("language_detection_unavailable_count") or pes2o.get("unigram_frequency_unavailable_count"):
        return "partial"
    return "usable"


def build_blockers(repair: dict, pes2o: dict, passage_stats: dict) -> list[str]:
    blockers = []
    if repair.get("still_missing_abstract_count"):
        blockers.append("metadata repair coverage: remaining missing abstracts require overrides")
    if pes2o.get("language_detection_unavailable_count"):
        blockers.append("language detection unavailable: pycld3 is not active")
    if pes2o.get("unigram_frequency_unavailable_count"):
        blockers.append("unigram frequency unavailable: section probability filter is skipped")
    if pes2o.get("bad_title_count") or pes2o.get("bad_fallback_title_count"):
        blockers.append("bad title leakage risk: review title repair and overrides")
    residue_count = sum(
        int(pes2o.get(key, 0))
        for key in [
            "reference_residue_count",
            "data_availability_residue_count",
            "ethics_residue_count",
            "supporting_information_residue_count",
        ]
    )
    if residue_count:
        blockers.append("residue leakage: non-textual or reference-like text remains")
    if passage_stats.get("not_index_ready_passage_count"):
        blockers.append("passage safety: non-index-ready records leaked into passages")
    blockers.append("streaming/batch limitations: passage loading and id checks need scalable versions")
    blockers.append("dedup/manifest limitations: long-term corpus updates need stronger provenance and dedup")
    return blockers[:5]


def build_markdown(report: dict) -> str:
    lines = ["# Pipeline Replication Review", ""]
    for key, title in [
        ("s2orc_replication_status", "S2ORC Replication Status"),
        ("pes2o_replication_status", "peS2o Replication Status"),
        ("openscholar_passage_status", "OpenScholar Passage Status"),
        ("chroma_indexing_status", "Chroma Indexing Status"),
    ]:
        lines.extend([f"## {title}", json.dumps(report[key], ensure_ascii=False, indent=2), ""])
    lines.extend(
        [
            "## Top Blockers Before 100K Scale",
            json.dumps(report["top_blockers_before_100k_scale"], ensure_ascii=False, indent=2),
            "",
        ]
    )
    return "\n".join(lines)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
