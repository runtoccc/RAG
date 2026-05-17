from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


DEFAULT_NODES = "data/agent/kg_nodes.jsonl"
DEFAULT_EDGES = "data/agent/kg_edges.jsonl"
DEFAULT_MD = "outputs/reports/kg_quality_report.md"
DEFAULT_JSON = "outputs/reports/kg_quality_report.json"

GENERIC_HUB_NAMES = {
    "growth",
    "methylation",
    "dna methylation",
    "reproduction",
    "disease",
    "infection",
    "temperature",
    "stress response",
    "immune response",
    "environmental factor",
    "pollution",
    "exposure",
    "epigenetic",
}


def main() -> None:
    args = parse_args()
    nodes_path = Path(args.nodes)
    edges_path = Path(args.edges)
    output_md = Path(args.output_md)
    output_json = Path(args.output_json)
    nodes = read_jsonl_required(nodes_path, "KG nodes")
    edges = read_jsonl_required(edges_path, "KG edges")

    report = build_report(nodes, edges)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(build_markdown(report), encoding="utf-8")

    print(f"[kg-quality] nodes={report['total_nodes']}")
    print(f"[kg-quality] edges={report['total_edges']}")
    print(f"[kg-quality] output_md={output_md}")
    print(f"[kg-quality] output_json={output_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a quality report for the local KG.")
    parser.add_argument("--nodes", default=DEFAULT_NODES)
    parser.add_argument("--edges", default=DEFAULT_EDGES)
    parser.add_argument("--output-md", default=DEFAULT_MD)
    parser.add_argument("--output-json", default=DEFAULT_JSON)
    return parser.parse_args()


def build_report(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    node_by_id = {node.get("entity_id"): node for node in nodes if node.get("entity_id")}
    node_type_distribution = Counter(node.get("entity_type") for node in nodes)
    predicate_distribution = Counter(edge.get("predicate") for edge in edges)
    degree = Counter()
    for edge in edges:
        degree.update([edge.get("subject_entity_id"), edge.get("object_entity_id")])
    isolated_nodes = [node for node in nodes if degree.get(node.get("entity_id"), 0) == 0]
    low_evidence_edges = [edge for edge in edges if int(edge.get("evidence_count") or 0) == 1]
    missing_doi_edges = [edge for edge in edges if not edge.get("dois")]
    missing_evidence_text_edges = [
        edge
        for edge in edges
        if not any((item.get("evidence_text") or "").strip() for item in edge.get("evidence") or [])
    ]
    generic_hubs = find_generic_hubs(nodes, edges, degree)
    recommendations = build_recommendations(
        nodes=nodes,
        edges=edges,
        node_type_distribution=node_type_distribution,
        predicate_distribution=predicate_distribution,
        low_evidence_edges=low_evidence_edges,
        missing_doi_edges=missing_doi_edges,
        missing_evidence_text_edges=missing_evidence_text_edges,
        generic_hubs=generic_hubs,
    )
    total_edges = len(edges)
    return {
        "warning": "KG edges are rule-extracted candidate relations. Treat them as evidence pointers, not manually verified biological facts.",
        "total_nodes": len(nodes),
        "total_edges": total_edges,
        "node_type_distribution": dict(node_type_distribution),
        "predicate_distribution": dict(predicate_distribution),
        "top_mention_nodes": top_mention_nodes(nodes, limit=20),
        "top_evidence_edges": top_evidence_edges(edges, limit=20),
        "low_evidence_edge_count": len(low_evidence_edges),
        "low_evidence_edge_ratio": safe_ratio(len(low_evidence_edges), total_edges),
        "missing_doi_edge_count": len(missing_doi_edges),
        "missing_doi_edge_ratio": safe_ratio(len(missing_doi_edges), total_edges),
        "missing_evidence_text_edge_count": len(missing_evidence_text_edges),
        "isolated_node_count": len(isolated_nodes),
        "possible_generic_hubs": generic_hubs,
        "recommendations": recommendations,
        "example_low_evidence_edges": [format_edge_summary(edge) for edge in low_evidence_edges[:10]],
        "example_missing_doi_edges": [format_edge_summary(edge) for edge in missing_doi_edges[:10]],
    }


def top_mention_nodes(nodes: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ranked = sorted(nodes, key=lambda node: int(node.get("mention_count") or 0), reverse=True)[:limit]
    return [
        {
            "entity_id": node.get("entity_id"),
            "name": node.get("normalized_name"),
            "entity_type": node.get("entity_type"),
            "mention_count": node.get("mention_count"),
            "paper_count": len(node.get("paper_ids") or []),
        }
        for node in ranked
    ]


def top_evidence_edges(edges: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ranked = sorted(edges, key=lambda edge: int(edge.get("evidence_count") or 0), reverse=True)[:limit]
    return [format_edge_summary(edge) for edge in ranked]


def format_edge_summary(edge: dict[str, Any]) -> dict[str, Any]:
    return {
        "edge_id": edge.get("edge_id"),
        "triple": f"{edge.get('subject')} -> {edge.get('predicate')} -> {edge.get('object')}",
        "evidence_count": edge.get("evidence_count"),
        "paper_count": len(edge.get("paper_ids") or []),
        "doi_count": len(edge.get("dois") or []),
        "max_confidence": edge.get("max_confidence"),
    }


def find_generic_hubs(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]], degree: Counter
) -> list[dict[str, Any]]:
    hub_threshold = max(8, int(len(edges) * 0.05))
    hubs = []
    for node in nodes:
        name = str(node.get("normalized_name") or "").lower()
        node_degree = int(degree.get(node.get("entity_id"), 0))
        if name in GENERIC_HUB_NAMES and node_degree >= hub_threshold:
            hubs.append(
                {
                    "entity_id": node.get("entity_id"),
                    "name": node.get("normalized_name"),
                    "entity_type": node.get("entity_type"),
                    "degree": node_degree,
                    "mention_count": node.get("mention_count"),
                    "flag": "possible_generic_hub",
                }
            )
    return sorted(hubs, key=lambda item: item["degree"], reverse=True)


def build_recommendations(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    node_type_distribution: Counter,
    predicate_distribution: Counter,
    low_evidence_edges: list[dict[str, Any]],
    missing_doi_edges: list[dict[str, Any]],
    missing_evidence_text_edges: list[dict[str, Any]],
    generic_hubs: list[dict[str, Any]],
) -> list[str]:
    recommendations = [
        "Do not present these KG edges as curated biological facts; present them as rule-extracted candidate evidence pointers.",
    ]
    expected_types = {"species", "gene", "protein", "phenotype", "chemical", "tissue", "method", "environmental_factor", "disease"}
    missing_or_sparse = [
        entity_type
        for entity_type in sorted(expected_types)
        if node_type_distribution.get(entity_type, 0) < 5
    ]
    if missing_or_sparse:
        recommendations.append(
            "Sparse entity types detected: "
            + ", ".join(missing_or_sparse)
            + ". Expand dictionaries or add LLM-assisted NER after creating QA samples."
        )
    if predicate_distribution:
        top_predicate, top_count = predicate_distribution.most_common(1)[0]
        if safe_ratio(top_count, len(edges)) > 0.4:
            recommendations.append(
                f"Predicate '{top_predicate}' dominates the graph. Relation extraction patterns may be too coarse."
            )
    if low_evidence_edges:
        recommendations.append(
            f"{len(low_evidence_edges)} edges have evidence_count == 1. Sample these manually before using them in answers."
        )
    if missing_doi_edges:
        recommendations.append(
            f"{len(missing_doi_edges)} edges have no DOI. This weakens citation-grounded answering."
        )
    if missing_evidence_text_edges:
        recommendations.append(
            f"{len(missing_evidence_text_edges)} edges have no evidence_text. These should not be shown as evidence."
        )
    if generic_hubs:
        recommendations.append(
            "Possible generic hubs detected: "
            + ", ".join(item["name"] for item in generic_hubs[:8])
            + ". Consider down-weighting generic nodes in retrieval."
        )
    if not missing_doi_edges and not missing_evidence_text_edges and len(low_evidence_edges) < max(5, len(edges) * 0.1):
        recommendations.append(
            "The KG is suitable for a prototype viewer and demo, but still needs sampled human QA before LLM-assisted relation extraction."
        )
    else:
        recommendations.append(
            "Before LLM-assisted relation extraction, create a small gold QA set and manually inspect low-evidence/generic-hub edges."
        )
    recommendations.append(
        "For PlantScience.ai-like progress, next prioritize KG QA, domain dictionaries, and evidence-aware relation extraction before adding complex GraphRAG."
    )
    return recommendations


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# KG Quality Report",
        "",
        f"> {report['warning']}",
        "",
        "## Summary",
        "",
        f"- total_nodes: {report['total_nodes']}",
        f"- total_edges: {report['total_edges']}",
        f"- low_evidence_edge_count: {report['low_evidence_edge_count']} ({report['low_evidence_edge_ratio']:.2%})",
        f"- missing_doi_edge_count: {report['missing_doi_edge_count']} ({report['missing_doi_edge_ratio']:.2%})",
        f"- missing_evidence_text_edge_count: {report['missing_evidence_text_edge_count']}",
        f"- isolated_node_count: {report['isolated_node_count']}",
        "",
        "## Node Type Distribution",
        "",
        dict_table(report["node_type_distribution"]),
        "",
        "## Predicate Distribution",
        "",
        dict_table(report["predicate_distribution"]),
        "",
        "## Top Mention Nodes",
        "",
        table(report["top_mention_nodes"], ["name", "entity_type", "mention_count", "paper_count"]),
        "",
        "## Top Evidence Edges",
        "",
        table(report["top_evidence_edges"], ["triple", "evidence_count", "paper_count", "doi_count", "max_confidence"]),
        "",
        "## Possible Generic Hubs",
        "",
        table(report["possible_generic_hubs"], ["name", "entity_type", "degree", "mention_count", "flag"])
        if report["possible_generic_hubs"]
        else "No high-degree generic hubs detected.",
        "",
        "## Recommendations",
        "",
    ]
    lines.extend(f"- {item}" for item in report["recommendations"])
    lines.append("")
    return "\n".join(lines)


def dict_table(values: dict[str, int]) -> str:
    rows = [{"name": key, "count": value} for key, value in sorted(values.items(), key=lambda item: item[1], reverse=True)]
    return table(rows, ["name", "count"])


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "No rows."
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(escape_md(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def read_jsonl_required(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    main()
