from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


DEFAULT_BUNDLE = "data/agent/evidence_bundle.json"
DEFAULT_REGISTRY = "data/agent/paper_registry.jsonl"
DEFAULT_OUTPUT = "data/agent/evidence_answer.json"


def main() -> None:
    configure_stdout()
    args = parse_args()
    bundle = json.loads(Path(args.bundle).read_text(encoding="utf-8"))
    registry = load_registry(Path(args.registry))
    answer = build_evidence_answer(bundle, registry=registry, max_claims=args.max_claims)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(answer, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[evidence-answer] output={output_path}")
    print(json.dumps(answer, ensure_ascii=False, indent=2))


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a citation-grounded answer skeleton from an evidence bundle.")
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-claims", type=int, default=5)
    return parser.parse_args()


def build_evidence_answer(
    bundle: dict[str, Any], registry: dict[str, dict[str, Any]], max_claims: int = 5
) -> dict[str, Any]:
    vector_evidence = bundle.get("vector_evidence") or []
    graph_evidence = bundle.get("graph_evidence") or []
    claims = build_claims(vector_evidence, graph_evidence, max_claims=max_claims)
    citations = build_citations(vector_evidence, graph_evidence, registry)
    return {
        "question": bundle.get("question"),
        "answer_mode": "evidence_skeleton_no_llm",
        "answer": build_short_answer(claims),
        "claims": claims,
        "citations": citations,
        "diagnostics": bundle.get("diagnostics") or {},
        "limitations": [
            "This is an extractive evidence skeleton, not a final LLM-written answer.",
            "Rule-extracted KG edges are candidates and should be treated as evidence pointers, not curated biological truth.",
            "Use passage_id and DOI to verify key claims before publication-quality use.",
        ],
    }


def build_claims(
    vector_evidence: list[dict[str, Any]],
    graph_evidence: list[dict[str, Any]],
    max_claims: int,
) -> list[dict[str, Any]]:
    claims = []
    for edge in graph_evidence:
        evidence_items = edge.get("evidence") or []
        passage_ids = unique_keep_order(
            [item.get("passage_id") for item in evidence_items if item.get("passage_id")]
        )
        claim = {
            "claim": f"{edge.get('subject')} {edge.get('predicate')} {edge.get('object')}",
            "claim_type": "graph_edge_candidate",
            "supporting_passage_ids": passage_ids,
            "supporting_edge_ids": [edge.get("edge_id")],
            "evidence_texts": [
                {
                    "passage_id": item.get("passage_id"),
                    "doi": item.get("doi"),
                    "text": item.get("evidence_text"),
                    "confidence": item.get("confidence"),
                }
                for item in evidence_items[:3]
            ],
            "confidence": confidence_label(edge),
        }
        claims.append(claim)
        if len(claims) >= max_claims:
            return claims

    for source in vector_evidence:
        claim = {
            "claim": f"Relevant passage retrieved from {source.get('title')}",
            "claim_type": "retrieved_passage",
            "supporting_passage_ids": [source.get("passage_id")],
            "supporting_edge_ids": [],
            "evidence_texts": [
                {
                    "passage_id": source.get("passage_id"),
                    "doi": source.get("doi"),
                    "text": source.get("snippet"),
                    "confidence": source.get("score"),
                }
            ],
            "confidence": "medium",
        }
        claims.append(claim)
        if len(claims) >= max_claims:
            break
    return claims


def build_citations(
    vector_evidence: list[dict[str, Any]],
    graph_evidence: list[dict[str, Any]],
    registry: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    citations: dict[tuple[str, str], dict[str, Any]] = {}
    for source in vector_evidence:
        key = (source.get("paper_id"), source.get("passage_id"))
        registry_row = registry.get(source.get("paper_id")) or {}
        citations[key] = {
            "paper_id": source.get("paper_id"),
            "title": source.get("title") or registry_row.get("title"),
            "doi": source.get("doi"),
            "passage_id": source.get("passage_id"),
            "source_file": source.get("source_file") or registry_row.get("source_file"),
        }
    for edge in graph_evidence:
        for item in edge.get("evidence") or []:
            key = (item.get("paper_id"), item.get("passage_id"))
            registry_row = registry.get(item.get("paper_id")) or {}
            citations.setdefault(
                key,
                {
                    "paper_id": item.get("paper_id"),
                    "title": registry_row.get("title"),
                    "doi": item.get("doi") or registry_row.get("doi"),
                    "passage_id": item.get("passage_id"),
                    "source_file": registry_row.get("source_file"),
                },
            )
    return list(citations.values())


def load_registry(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    registry = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("paper_id"):
                registry[row["paper_id"]] = row
    return registry


def build_short_answer(claims: list[dict[str, Any]]) -> str:
    if not claims:
        return "当前 evidence bundle 没有足够证据形成回答。"
    lines = ["基于当前检索证据，可以先形成以下可核查结论草稿："]
    for index, claim in enumerate(claims, start=1):
        passage_ids = ", ".join(claim.get("supporting_passage_ids") or [])
        lines.append(f"{index}. {claim['claim']}。证据 passage: {passage_ids}")
    return "\n".join(lines)


def confidence_label(edge: dict[str, Any]) -> str:
    evidence_count = int(edge.get("evidence_count") or 0)
    score = float(edge.get("score") or 0.0)
    if evidence_count >= 5 and score >= 120:
        return "medium"
    if evidence_count >= 2:
        return "low-medium"
    return "low"


def unique_keep_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


if __name__ == "__main__":
    main()
