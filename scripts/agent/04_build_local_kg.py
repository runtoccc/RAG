from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterator

import networkx as nx


DEFAULT_ENTITIES = "data/agent/entity_candidates.jsonl"
DEFAULT_RELATIONS = "data/agent/relation_candidates.jsonl"
DEFAULT_NODES = "data/agent/kg_nodes.jsonl"
DEFAULT_EDGES = "data/agent/kg_edges.jsonl"
DEFAULT_GRAPHML = "data/agent/local_kg.graphml"


def main() -> None:
    args = parse_args()
    entities_path = Path(args.entities)
    relations_path = Path(args.relations)
    nodes_path = Path(args.nodes)
    edges_path = Path(args.edges)
    graphml_path = Path(args.graphml)
    for path in [nodes_path, edges_path, graphml_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    nodes = build_nodes(entities_path)
    edges = build_edges(relations_path)
    write_jsonl(nodes_path, nodes.values())
    write_jsonl(edges_path, edges.values())
    graph = build_graph(nodes, edges)
    nx.write_graphml(graph, graphml_path)

    print(f"[local-kg] nodes={len(nodes)}")
    print(f"[local-kg] edges={len(edges)}")
    print(f"[local-kg] node_types={dict(Counter(node['entity_type'] for node in nodes.values()))}")
    print(f"[local-kg] predicates={dict(Counter(edge['predicate'] for edge in edges.values()))}")
    print(f"[local-kg] nodes_output={nodes_path}")
    print(f"[local-kg] edges_output={edges_path}")
    print(f"[local-kg] graphml_output={graphml_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local fish/aquaculture KG from candidates.")
    parser.add_argument("--entities", default=DEFAULT_ENTITIES)
    parser.add_argument("--relations", default=DEFAULT_RELATIONS)
    parser.add_argument("--nodes", default=DEFAULT_NODES)
    parser.add_argument("--edges", default=DEFAULT_EDGES)
    parser.add_argument("--graphml", default=DEFAULT_GRAPHML)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_nodes(path: Path) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for entity in iter_jsonl(path):
        entity_id = entity["entity_id"]
        node = grouped.setdefault(
            entity_id,
            {
                "entity_id": entity_id,
                "normalized_name": entity["normalized_name"],
                "entity_type": entity["entity_type"],
                "surfaces": [],
                "paper_ids": [],
                "passage_ids": [],
                "source_files": [],
                "dois": [],
                "mention_count": 0,
                "max_confidence": 0.0,
            },
        )
        node["mention_count"] += 1
        node["max_confidence"] = max(node["max_confidence"], float(entity.get("confidence") or 0.0))
        append_unique(node["surfaces"], entity.get("surface"))
        append_unique(node["paper_ids"], entity.get("paper_id"))
        append_unique(node["passage_ids"], entity.get("passage_id"))
        append_unique(node["source_files"], entity.get("source_file"))
        append_unique(node["dois"], entity.get("doi"))
    return grouped


def build_edges(path: Path) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for relation in iter_jsonl(path):
        edge_key = make_edge_key(relation)
        edge = grouped.setdefault(
            edge_key,
            {
                "edge_id": edge_key,
                "subject_entity_id": relation.get("subject_entity_id"),
                "subject": relation.get("subject"),
                "subject_type": relation.get("subject_type"),
                "predicate": relation.get("predicate"),
                "object_entity_id": relation.get("object_entity_id"),
                "object": relation.get("object"),
                "object_type": relation.get("object_type"),
                "paper_ids": [],
                "passage_ids": [],
                "dois": [],
                "evidence_count": 0,
                "max_confidence": 0.0,
                "extractors": [],
                "evidence": [],
            },
        )
        edge["evidence_count"] += 1
        edge["max_confidence"] = max(edge["max_confidence"], float(relation.get("confidence") or 0.0))
        append_unique(edge["paper_ids"], relation.get("paper_id"))
        append_unique(edge["passage_ids"], relation.get("passage_id"))
        append_unique(edge["dois"], relation.get("doi"))
        append_unique(edge["extractors"], relation.get("extractor"))
        if len(evidence[edge_key]) < 5:
            evidence[edge_key].append(
                {
                    "paper_id": relation.get("paper_id"),
                    "passage_id": relation.get("passage_id"),
                    "doi": relation.get("doi"),
                    "evidence_text": relation.get("evidence_text"),
                    "confidence": relation.get("confidence"),
                }
            )
    for edge_key, edge_evidence in evidence.items():
        grouped[edge_key]["evidence"] = edge_evidence
    return grouped


def make_edge_key(relation: dict[str, Any]) -> str:
    subject_id = relation.get("subject_entity_id") or relation.get("subject")
    object_id = relation.get("object_entity_id") or relation.get("object")
    return f"{subject_id}::{relation.get('predicate')}::{object_id}"


def build_graph(nodes: dict[str, dict[str, Any]], edges: dict[str, dict[str, Any]]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for node_id, node in nodes.items():
        graph.add_node(
            node_id,
            label=node["normalized_name"],
            entity_type=node["entity_type"],
            mention_count=node["mention_count"],
            paper_count=len(node["paper_ids"]),
            max_confidence=node["max_confidence"],
        )
    for edge_id, edge in edges.items():
        subject_id = edge.get("subject_entity_id")
        object_id = edge.get("object_entity_id")
        if not subject_id or not object_id or subject_id not in nodes or object_id not in nodes:
            continue
        graph.add_edge(
            subject_id,
            object_id,
            key=edge_id,
            predicate=edge["predicate"],
            evidence_count=edge["evidence_count"],
            paper_count=len(edge["paper_ids"]),
            max_confidence=edge["max_confidence"],
        )
    return graph


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_unique(values: list[Any], value: Any) -> None:
    if value and value not in values:
        values.append(value)


if __name__ == "__main__":
    main()
