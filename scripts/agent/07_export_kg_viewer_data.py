from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_NODES = "data/agent/kg_nodes.jsonl"
DEFAULT_EDGES = "data/agent/kg_edges.jsonl"
DEFAULT_BUNDLE = "data/agent/evidence_bundle.json"
DEFAULT_REGISTRY = "data/agent/paper_registry.jsonl"
DEFAULT_OUTPUT = "data/agent/kg_viewer_data.json"


def main() -> None:
    args = parse_args()
    nodes_path = Path(args.nodes)
    edges_path = Path(args.edges)
    bundle_path = Path(args.bundle)
    registry_path = Path(args.registry)
    output_path = Path(args.output)

    nodes = read_jsonl_required(nodes_path, "KG nodes")
    edges = read_jsonl_required(edges_path, "KG edges")
    bundle = read_json_optional(bundle_path)
    registry = read_registry(registry_path)

    viewer_data = build_viewer_data(
        nodes=nodes,
        edges=edges,
        bundle=bundle,
        registry=registry,
        nodes_path=nodes_path,
        edges_path=edges_path,
        bundle_path=bundle_path,
        registry_path=registry_path,
        mode=args.mode,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        min_edge_evidence=args.min_edge_evidence,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(viewer_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[kg-viewer-data] nodes_input={len(nodes)}")
    print(f"[kg-viewer-data] edges_input={len(edges)}")
    print(f"[kg-viewer-data] exported_nodes={len(viewer_data['nodes'])}")
    print(f"[kg-viewer-data] exported_edges={len(viewer_data['edges'])}")
    print(f"[kg-viewer-data] output={output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export local KG data for the HTML viewer.")
    parser.add_argument("--nodes", default=DEFAULT_NODES)
    parser.add_argument("--edges", default=DEFAULT_EDGES)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", default="retrieved_or_top", choices=["retrieved_or_top", "top"])
    parser.add_argument("--max-nodes", type=int, default=150)
    parser.add_argument("--max-edges", type=int, default=250)
    parser.add_argument("--min-edge-evidence", type=int, default=1)
    return parser.parse_args()


def build_viewer_data(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    bundle: dict[str, Any] | None,
    registry: dict[str, dict[str, Any]],
    nodes_path: Path,
    edges_path: Path,
    bundle_path: Path,
    registry_path: Path,
    mode: str,
    max_nodes: int,
    max_edges: int,
    min_edge_evidence: int,
) -> dict[str, Any]:
    node_by_id = {node["entity_id"]: node for node in nodes if node.get("entity_id")}
    edge_by_id = {edge["edge_id"]: edge for edge in edges if edge.get("edge_id")}
    selected_edge_ids: list[str] = []
    selected_node_ids: list[str] = []
    matched_node_ids = set()
    retrieved_edge_ids = set()

    if bundle and mode == "retrieved_or_top":
        matched_node_ids = {
            node.get("entity_id")
            for node in bundle.get("matched_nodes", [])
            if node.get("entity_id")
        }
        retrieved_edge_ids = {
            edge.get("edge_id")
            for edge in bundle.get("graph_evidence", [])
            if edge.get("edge_id")
        }
        selected_node_ids.extend([node_id for node_id in matched_node_ids if node_id in node_by_id])
        selected_edge_ids.extend([edge_id for edge_id in retrieved_edge_ids if edge_id in edge_by_id])

    eligible_edges = [
        edge
        for edge in edges
        if int(edge.get("evidence_count") or 0) >= min_edge_evidence
        and edge.get("subject_entity_id") in node_by_id
        and edge.get("object_entity_id") in node_by_id
    ]
    top_edges = sorted(
        eligible_edges,
        key=lambda edge: (
            int(edge.get("evidence_count") or 0),
            len(edge.get("paper_ids") or []),
            float(edge.get("max_confidence") or 0.0),
        ),
        reverse=True,
    )

    if len(selected_edge_ids) < min(max_edges, 20) or not selected_edge_ids:
        for edge in top_edges:
            append_unique(selected_edge_ids, edge["edge_id"])
            if len(selected_edge_ids) >= max_edges:
                break

    exported_edges_raw: list[dict[str, Any]] = []
    exported_node_ids = set(selected_node_ids)
    for edge_id in selected_edge_ids:
        edge = edge_by_id.get(edge_id)
        if not edge:
            continue
        subject_id = edge.get("subject_entity_id")
        object_id = edge.get("object_entity_id")
        if not subject_id or not object_id:
            continue
        prospective_nodes = {subject_id, object_id} - exported_node_ids
        if len(exported_node_ids) + len(prospective_nodes) > max_nodes:
            continue
        exported_node_ids.update([subject_id, object_id])
        exported_edges_raw.append(edge)
        if len(exported_edges_raw) >= max_edges:
            break

    for node_id in selected_node_ids:
        if node_id in node_by_id and len(exported_node_ids) < max_nodes:
            exported_node_ids.add(node_id)

    exported_nodes_raw = [node_by_id[node_id] for node_id in sorted(exported_node_ids) if node_id in node_by_id]
    exported_nodes = [
        format_node(node, registry=registry, is_query_matched=node["entity_id"] in matched_node_ids)
        for node in exported_nodes_raw
    ]
    exported_edges = [
        format_edge(edge, registry=registry, is_retrieved=edge["edge_id"] in retrieved_edge_ids)
        for edge in exported_edges_raw
    ]
    stats = build_stats(nodes, edges, exported_nodes, exported_edges)

    return {
        "nodes": exported_nodes,
        "edges": exported_edges,
        "stats": stats,
        "source": {
            "nodes_path": str(nodes_path),
            "edges_path": str(edges_path),
            "bundle_path": str(bundle_path) if bundle_path.exists() else None,
            "registry_path": str(registry_path) if registry_path.exists() else None,
            "mode": mode,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
    }


def format_node(node: dict[str, Any], registry: dict[str, dict[str, Any]], is_query_matched: bool) -> dict[str, Any]:
    paper_ids = node.get("paper_ids") or []
    source_files = unique_keep_order(
        list(node.get("source_files") or [])
        + [registry.get(paper_id, {}).get("source_file") for paper_id in paper_ids]
    )
    return {
        "id": node.get("entity_id"),
        "label": node.get("normalized_name"),
        "entity_type": node.get("entity_type"),
        "mention_count": int(node.get("mention_count") or 0),
        "paper_count": len(paper_ids),
        "max_confidence": float(node.get("max_confidence") or 0.0),
        "paper_ids": paper_ids,
        "passage_ids": node.get("passage_ids") or [],
        "dois": node.get("dois") or [],
        "source_files": source_files,
        "is_query_matched": is_query_matched,
        "group": node.get("entity_type"),
    }


def format_edge(edge: dict[str, Any], registry: dict[str, dict[str, Any]], is_retrieved: bool) -> dict[str, Any]:
    evidence = []
    for item in (edge.get("evidence") or [])[:5]:
        paper_id = item.get("paper_id")
        registry_row = registry.get(paper_id) or {}
        evidence.append(
            {
                "paper_id": paper_id,
                "passage_id": item.get("passage_id"),
                "doi": item.get("doi") or registry_row.get("doi"),
                "source_file": registry_row.get("source_file"),
                "title": registry_row.get("title"),
                "evidence_text": item.get("evidence_text"),
                "confidence": item.get("confidence"),
            }
        )
    return {
        "id": edge.get("edge_id"),
        "source": edge.get("subject_entity_id"),
        "target": edge.get("object_entity_id"),
        "label": edge.get("predicate"),
        "predicate": edge.get("predicate"),
        "subject": edge.get("subject"),
        "object": edge.get("object"),
        "subject_type": edge.get("subject_type"),
        "object_type": edge.get("object_type"),
        "evidence_count": int(edge.get("evidence_count") or 0),
        "paper_count": len(edge.get("paper_ids") or []),
        "max_confidence": float(edge.get("max_confidence") or 0.0),
        "paper_ids": edge.get("paper_ids") or [],
        "passage_ids": edge.get("passage_ids") or [],
        "dois": edge.get("dois") or [],
        "evidence": evidence,
        "is_retrieved": is_retrieved,
    }


def build_stats(
    all_nodes: list[dict[str, Any]],
    all_edges: list[dict[str, Any]],
    exported_nodes: list[dict[str, Any]],
    exported_edges: list[dict[str, Any]],
) -> dict[str, Any]:
    connected = set()
    for edge in exported_edges:
        connected.add(edge.get("source"))
        connected.add(edge.get("target"))
    isolated = [node for node in exported_nodes if node.get("id") not in connected]
    return {
        "total_nodes_input": len(all_nodes),
        "total_edges_input": len(all_edges),
        "exported_nodes": len(exported_nodes),
        "exported_edges": len(exported_edges),
        "node_type_distribution": dict(Counter(node.get("entity_type") for node in exported_nodes)),
        "predicate_distribution": dict(Counter(edge.get("predicate") for edge in exported_edges)),
        "missing_doi_edge_count": sum(1 for edge in exported_edges if not edge.get("dois")),
        "low_evidence_edge_count": sum(1 for edge in exported_edges if int(edge.get("evidence_count") or 0) <= 1),
        "isolated_node_count_in_export": len(isolated),
    }


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


def read_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        print(f"[kg-viewer-data] optional bundle not found: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_registry(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        print(f"[kg-viewer-data] optional registry not found: {path}")
        return {}
    registry = {}
    for row in read_jsonl_required(path, "paper registry"):
        if row.get("paper_id"):
            registry[row["paper_id"]] = row
    return registry


def append_unique(values: list[str], value: str | None) -> None:
    if value and value not in values:
        values.append(value)


def unique_keep_order(values: Iterable[Any]) -> list[Any]:
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
