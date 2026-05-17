from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
from difflib import SequenceMatcher
import sys
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - fallback for non-rag shells
    fuzz = None


DEFAULT_NODES = "data/agent/kg_nodes.jsonl"
DEFAULT_EDGES = "data/agent/kg_edges.jsonl"
DEFAULT_OUTPUT = "data/agent/evidence_bundle.json"


def main() -> None:
    configure_stdout()
    args = parse_args()
    bundle = hybrid_retrieve(
        question=args.question,
        top_k=args.top_k,
        node_k=args.node_k,
        edge_k=args.edge_k,
        nodes_path=Path(args.nodes),
        edges_path=Path(args.edges),
        skip_vector=args.skip_vector,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[hybrid-retrieve] output={output_path}")
    print(json.dumps(bundle, ensure_ascii=False, indent=2))


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a vector + KG evidence bundle for the literature agent.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--node-k", type=int, default=8)
    parser.add_argument("--edge-k", type=int, default=12)
    parser.add_argument("--nodes", default=DEFAULT_NODES)
    parser.add_argument("--edges", default=DEFAULT_EDGES)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-vector", action="store_true")
    return parser.parse_args()


def hybrid_retrieve(
    question: str,
    top_k: int,
    node_k: int,
    edge_k: int,
    nodes_path: Path,
    edges_path: Path,
    skip_vector: bool = False,
) -> dict[str, Any]:
    nodes = list(iter_jsonl(nodes_path))
    edges = list(iter_jsonl(edges_path))
    query_terms = extract_query_terms(question)
    matched_nodes = match_nodes(question, query_terms, nodes, limit=node_k)
    graph_edges = retrieve_graph_edges(query_terms, matched_nodes, edges, limit=edge_k)
    vector_sources, vector_error = retrieve_vector_sources(question, top_k, skip_vector=skip_vector)
    linked_passage_ids = collect_linked_passage_ids(vector_sources, graph_edges)

    return {
        "question": question,
        "query_terms": query_terms,
        "vector_error": vector_error,
        "vector_evidence": vector_sources,
        "matched_nodes": matched_nodes,
        "graph_evidence": graph_edges,
        "linked_passage_ids": linked_passage_ids,
        "diagnostics": {
            "vector_evidence_count": len(vector_sources),
            "matched_node_count": len(matched_nodes),
            "graph_edge_count": len(graph_edges),
            "matched_node_types": dict(Counter(node["entity_type"] for node in matched_nodes)),
            "graph_predicates": dict(Counter(edge["predicate"] for edge in graph_edges)),
        },
        "answer_contract": {
            "use_only_evidence": True,
            "cite_passage_ids": True,
            "cite_edge_ids_when_using_graph": True,
            "say_insufficient_evidence_when_needed": True,
        },
    }


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_query_terms(question: str) -> list[str]:
    terms = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", question):
        terms.append(token.lower())
    domain_terms = {
        "methyl": ["methylation", "dna methylation", "epigenetic"],
        "dnmt": ["dnmt1", "dnmt3", "dna methylation"],
        "cyp19a": ["cyp19a", "aromatase", "sex differentiation"],
        "aromatase": ["cyp19a", "aromatase"],
        "zebrafish": ["zebrafish", "danio rerio"],
        "tilapia": ["nile tilapia", "oreochromis niloticus"],
        "exposure": ["exposure", "xenobiotic", "pollutant"],
        "pollutant": ["pollutant", "xenobiotic", "exposure"],
        "temperature": ["temperature", "sex reversal", "sex determination"],
        "growth": ["growth", "muscle"],
    }
    lower_question = question.lower()
    for trigger, additions in domain_terms.items():
        if trigger in lower_question:
            terms.extend(additions)
    return unique_keep_order(terms)


def match_nodes(
    question: str, query_terms: list[str], nodes: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    scored = []
    question_lower = question.lower()
    query_text = " ".join(query_terms) or question_lower
    for node in nodes:
        name = str(node.get("normalized_name") or "")
        surfaces = node.get("surfaces") or []
        aliases = [name, *[str(surface) for surface in surfaces]]
        exact = any(alias.lower() in question_lower for alias in aliases if alias)
        fuzzy = max((fuzzy_score(query_text, alias.lower()) for alias in aliases if alias), default=0)
        score = (100 if exact else 0) + fuzzy + min(int(node.get("mention_count") or 0), 30) * 0.5
        if exact or fuzzy >= 72:
            scored.append((score, node))
    ranked = sorted(scored, key=lambda item: item[0], reverse=True)[:limit]
    return [format_node(node, score) for score, node in ranked]


def format_node(node: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "entity_id": node.get("entity_id"),
        "normalized_name": node.get("normalized_name"),
        "entity_type": node.get("entity_type"),
        "score": round(float(score), 3),
        "mention_count": node.get("mention_count"),
        "paper_count": len(node.get("paper_ids") or []),
        "paper_ids": (node.get("paper_ids") or [])[:10],
        "passage_ids": (node.get("passage_ids") or [])[:10],
        "dois": (node.get("dois") or [])[:10],
    }


def retrieve_graph_edges(
    query_terms: list[str],
    matched_nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    node_ids = {node["entity_id"] for node in matched_nodes}
    query_text = " ".join(query_terms)
    scored = []
    for edge in edges:
        subject_id = edge.get("subject_entity_id")
        object_id = edge.get("object_entity_id")
        touches_node = subject_id in node_ids or object_id in node_ids
        edge_text = " ".join(
            str(edge.get(field) or "")
            for field in ["subject", "predicate", "object", "subject_type", "object_type"]
        ).lower()
        term_hits = sum(1 for term in query_terms if term and term.lower() in edge_text)
        fuzzy = fuzzy_score(query_text, edge_text) if query_text else 0
        if not touches_node and term_hits == 0 and fuzzy < 70:
            continue
        score = (80 if touches_node else 0) + term_hits * 12 + fuzzy * 0.5
        score += min(int(edge.get("evidence_count") or 0), 10) * 1.5
        scored.append((score, edge))
    ranked = sorted(scored, key=lambda item: item[0], reverse=True)[:limit]
    return [format_edge(edge, score) for score, edge in ranked]


def format_edge(edge: dict[str, Any], score: float) -> dict[str, Any]:
    evidence = edge.get("evidence") or []
    return {
        "edge_id": edge.get("edge_id"),
        "subject": edge.get("subject"),
        "subject_type": edge.get("subject_type"),
        "predicate": edge.get("predicate"),
        "object": edge.get("object"),
        "object_type": edge.get("object_type"),
        "score": round(float(score), 3),
        "evidence_count": edge.get("evidence_count"),
        "paper_ids": (edge.get("paper_ids") or [])[:10],
        "passage_ids": (edge.get("passage_ids") or [])[:10],
        "dois": (edge.get("dois") or [])[:10],
        "evidence": evidence[:3],
    }


def retrieve_vector_sources(question: str, top_k: int, skip_vector: bool = False) -> tuple[list[dict[str, Any]], str | None]:
    if skip_vector:
        return [], "vector retrieval skipped by user"
    try:
        from query_data import retrieve_chunks

        sources = retrieve_chunks(question, top_k=top_k)
    except Exception as error:
        return [], f"{error.__class__.__name__}: {error}"
    return [format_vector_source(source, index) for index, source in enumerate(sources, start=1)], None


def fuzzy_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if fuzz is not None:
        return float(fuzz.token_set_ratio(left, right))
    return SequenceMatcher(None, left, right).ratio() * 100


def format_vector_source(source: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "source_id": f"S{index}",
        "paper_id": source.get("paper_id"),
        "passage_id": source.get("chunk_id"),
        "title": source.get("title"),
        "doi": source.get("doi"),
        "source_file": source.get("source_file") or source.get("pdf_file"),
        "score": source.get("score"),
        "retrieval_method": source.get("retrieval_method"),
        "matched_terms": source.get("matched_terms") or [],
        "snippet": source.get("snippet"),
    }


def collect_linked_passage_ids(
    vector_sources: list[dict[str, Any]], graph_edges: list[dict[str, Any]]
) -> list[str]:
    passage_ids = []
    for source in vector_sources:
        passage_ids.append(source.get("passage_id"))
    for edge in graph_edges:
        passage_ids.extend(edge.get("passage_ids") or [])
    return unique_keep_order([pid for pid in passage_ids if pid])


def unique_keep_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


if __name__ == "__main__":
    main()
