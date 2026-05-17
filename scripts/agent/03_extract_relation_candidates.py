from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterator


DEFAULT_PASSAGES = "data/passages/openscholar_passages.jsonl"
DEFAULT_COMPAT_PASSAGES = "data/passages/scientific_passages.jsonl"
DEFAULT_ENTITIES = "data/agent/entity_candidates.jsonl"
DEFAULT_OUTPUT = "data/agent/relation_candidates.jsonl"

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

RELATION_PATTERNS = [
    ("upregulates", re.compile(r"\b(up-?regulat(?:es|ed|ion)|increase[sd]?|elevate[sd]?)\b", re.I), 0.68),
    ("downregulates", re.compile(r"\b(down-?regulat(?:es|ed|ion)|decrease[sd]?|reduce[sd]?|suppress(?:es|ed)?)\b", re.I), 0.68),
    ("regulates", re.compile(r"\b(regulat(?:es|ed|ion)|modulat(?:es|ed|ion)|control(?:s|led)?)\b", re.I), 0.66),
    ("affects", re.compile(r"\b(affect(?:s|ed)?|alter(?:s|ed)?|influence[sd]?|impact(?:s|ed)?|induce[sd]?)\b", re.I), 0.64),
    ("associated_with", re.compile(r"\b(associated with|related to|linked to|involved in|correlated with)\b", re.I), 0.62),
    ("expressed_in", re.compile(r"\b(express(?:ed|ion)? in|express(?:ed|ion)? levels? in)\b", re.I), 0.7),
    ("measured_by", re.compile(r"\b(measured by|detected by|analy[sz]ed by|using|performed by)\b", re.I), 0.58),
]

TYPE_PAIR_PRIORITIES = {
    "expressed_in": [("gene", "tissue"), ("protein", "tissue")],
    "measured_by": [
        ("gene", "method"),
        ("protein", "method"),
        ("phenotype", "method"),
        ("chemical", "method"),
    ],
    "observed_in_species": [
        ("gene", "species"),
        ("protein", "species"),
        ("phenotype", "species"),
        ("chemical", "species"),
        ("environmental_factor", "species"),
        ("tissue", "species"),
    ],
}

PREDICATE_ALLOWED_PAIRS = {
    "upregulates": {
        ("gene", "gene"),
        ("gene", "protein"),
        ("protein", "gene"),
        ("chemical", "gene"),
        ("chemical", "phenotype"),
        ("environmental_factor", "gene"),
        ("environmental_factor", "phenotype"),
        ("phenotype", "gene"),
        ("phenotype", "phenotype"),
    },
    "downregulates": {
        ("gene", "gene"),
        ("gene", "protein"),
        ("protein", "gene"),
        ("chemical", "gene"),
        ("chemical", "phenotype"),
        ("environmental_factor", "gene"),
        ("environmental_factor", "phenotype"),
        ("phenotype", "gene"),
        ("phenotype", "phenotype"),
    },
    "regulates": {
        ("gene", "gene"),
        ("gene", "protein"),
        ("protein", "gene"),
        ("chemical", "gene"),
        ("chemical", "phenotype"),
        ("environmental_factor", "gene"),
        ("environmental_factor", "phenotype"),
        ("phenotype", "gene"),
        ("phenotype", "phenotype"),
    },
    "affects": {
        ("chemical", "gene"),
        ("chemical", "protein"),
        ("chemical", "phenotype"),
        ("chemical", "tissue"),
        ("environmental_factor", "gene"),
        ("environmental_factor", "protein"),
        ("environmental_factor", "phenotype"),
        ("environmental_factor", "tissue"),
        ("phenotype", "gene"),
        ("phenotype", "phenotype"),
    },
    "associated_with": {
        ("gene", "phenotype"),
        ("gene", "tissue"),
        ("protein", "phenotype"),
        ("chemical", "phenotype"),
        ("environmental_factor", "phenotype"),
        ("phenotype", "gene"),
        ("phenotype", "protein"),
        ("phenotype", "tissue"),
        ("phenotype", "phenotype"),
    },
    "expressed_in": {("gene", "tissue"), ("protein", "tissue")},
    "measured_by": {
        ("gene", "method"),
        ("protein", "method"),
        ("phenotype", "method"),
        ("chemical", "method"),
        ("tissue", "method"),
    },
}


def main() -> None:
    args = parse_args()
    passages_path = resolve_passages(Path(args.passages))
    entities_path = Path(args.entities)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    passages = {p["passage_id"]: p for p in iter_jsonl(passages_path)}
    entities_by_passage = load_entities_by_passage(entities_path)
    seen: set[str] = set()
    relation_count = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for passage_id, entities in entities_by_passage.items():
            passage = passages.get(passage_id)
            if not passage:
                continue
            for relation in extract_relations_for_passage(passage, entities):
                if relation["relation_id"] in seen:
                    continue
                seen.add(relation["relation_id"])
                relation_count += 1
                output_file.write(json.dumps(relation, ensure_ascii=False) + "\n")

    print(f"[relation-candidates] passages={len(passages)}")
    print(f"[relation-candidates] entity_passages={len(entities_by_passage)}")
    print(f"[relation-candidates] relations={relation_count}")
    print(f"[relation-candidates] output={output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract evidence-backed relation candidates from entity candidates.")
    parser.add_argument("--passages", default=DEFAULT_PASSAGES)
    parser.add_argument("--entities", default=DEFAULT_ENTITIES)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve_passages(path: Path) -> Path:
    if path.exists():
        return path
    compat = Path(DEFAULT_COMPAT_PASSAGES)
    if path == Path(DEFAULT_PASSAGES) and compat.exists():
        return compat
    raise FileNotFoundError(path)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_entities_by_passage(path: Path) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entity in iter_jsonl(path):
        passage_id = entity.get("passage_id")
        if passage_id:
            groups[passage_id].append(entity)
    return groups


def extract_relations_for_passage(passage: dict[str, Any], entities: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    unique_entities = dedupe_entities(entities)
    if len(unique_entities) < 2:
        return

    text = passage.get("text") or ""
    for sentence in split_sentences(text):
        sentence_entities = entities_in_sentence(sentence, unique_entities)
        if len(sentence_entities) < 2:
            continue
        for predicate, pattern, confidence in RELATION_PATTERNS:
            match = pattern.search(sentence)
            if not match:
                continue
            pair = choose_entity_pair(predicate, sentence, match.start(), sentence_entities)
            if not pair:
                continue
            subject, obj = pair
            yield make_relation(
                passage,
                subject,
                obj,
                predicate,
                evidence_text=sentence,
                extractor="pattern_sentence_v1",
                confidence=confidence,
            )

    yield from extract_observed_in_species_relations(passage, text, unique_entities)


def dedupe_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for entity in entities:
        key = (entity.get("entity_type"), entity.get("normalized_name"))
        if key in seen:
            continue
        seen.add(key)
        result.append(entity)
    return result


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.split(text) if 30 <= len(s.strip()) <= 900]


def entities_in_sentence(sentence: str, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lowered = sentence.lower()
    found = []
    for entity in entities:
        name = str(entity.get("normalized_name") or "").lower()
        surface = str(entity.get("surface") or "").lower()
        positions = [pos for value in {name, surface} if value for pos in [lowered.find(value)] if pos >= 0]
        if not positions:
            continue
        copy = dict(entity)
        copy["_pos"] = min(positions)
        found.append(copy)
    return sorted(found, key=lambda item: item["_pos"])


def choose_entity_pair(
    predicate: str, sentence: str, predicate_pos: int, entities: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    allowed_pairs = PREDICATE_ALLOWED_PAIRS.get(predicate)
    preferred = TYPE_PAIR_PRIORITIES.get(predicate)
    if preferred:
        for left_type, right_type in preferred:
            left = first_entity_of_type(entities, left_type)
            right = first_entity_of_type(entities, right_type)
            if left and right and left["normalized_name"] != right["normalized_name"]:
                return left, right

    before = [e for e in entities if e["_pos"] <= predicate_pos]
    after = [e for e in entities if e["_pos"] > predicate_pos]
    if before and after:
        pair = choose_allowed_pair(before[::-1], after, allowed_pairs)
        if pair:
            return pair

    pair = choose_allowed_pair(entities, entities, allowed_pairs)
    if pair:
        return pair
    return None


def choose_allowed_pair(
    left_entities: list[dict[str, Any]],
    right_entities: list[dict[str, Any]],
    allowed_pairs: set[tuple[str, str]] | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    for left in left_entities:
        for right in right_entities:
            if left is right or left.get("normalized_name") == right.get("normalized_name"):
                continue
            if names_are_nested(left.get("normalized_name"), right.get("normalized_name")):
                continue
            pair = (left.get("entity_type"), right.get("entity_type"))
            if allowed_pairs is None or pair in allowed_pairs:
                return left, right
    return None


def names_are_nested(left_name: Any, right_name: Any) -> bool:
    left = str(left_name or "").strip().lower()
    right = str(right_name or "").strip().lower()
    if not left or not right:
        return False
    return left in right or right in left


def first_entity_of_type(entities: list[dict[str, Any]], entity_type: str) -> dict[str, Any] | None:
    for entity in entities:
        if entity.get("entity_type") == entity_type:
            return entity
    return None


def extract_observed_in_species_relations(
    passage: dict[str, Any], text: str, entities: list[dict[str, Any]]
) -> Iterator[dict[str, Any]]:
    species_entities = [e for e in entities if e.get("entity_type") == "species"]
    if not species_entities:
        return
    species = species_entities[0]
    for entity in entities:
        if entity.get("entity_type") == "species":
            continue
        if entity.get("entity_type") not in {"gene", "protein", "phenotype", "chemical", "environmental_factor"}:
            continue
        evidence = shared_evidence(entity, species, text)
        if not evidence:
            continue
        yield make_relation(
            passage,
            entity,
            species,
            "observed_in_species",
            evidence_text=evidence,
            extractor="cooccurrence_species_v1",
            confidence=0.46,
        )


def shared_evidence(entity: dict[str, Any], species: dict[str, Any], text: str) -> str | None:
    entity_name = str(entity.get("normalized_name") or "").lower()
    species_name = str(species.get("normalized_name") or "").lower()
    for sentence in split_sentences(text):
        lowered = sentence.lower()
        if entity_name in lowered and species_name in lowered:
            return sentence
    return None


def make_relation(
    passage: dict[str, Any],
    subject: dict[str, Any],
    obj: dict[str, Any],
    predicate: str,
    evidence_text: str,
    extractor: str,
    confidence: float,
) -> dict[str, Any]:
    paper_id = passage.get("paper_id") or ""
    passage_id = passage.get("passage_id") or ""
    subject_name = subject.get("normalized_name") or subject.get("surface") or ""
    object_name = obj.get("normalized_name") or obj.get("surface") or ""
    relation_id = stable_id("rel", paper_id, passage_id, str(subject_name), predicate, str(object_name))
    return {
        "relation_id": relation_id,
        "subject": subject_name,
        "subject_type": subject.get("entity_type"),
        "subject_entity_id": subject.get("entity_id"),
        "predicate": predicate,
        "object": object_name,
        "object_type": obj.get("entity_type"),
        "object_entity_id": obj.get("entity_id"),
        "paper_id": paper_id,
        "passage_id": passage_id,
        "source_file": passage.get("source_file"),
        "doi": passage.get("doi"),
        "evidence_text": " ".join(evidence_text.split()),
        "extractor": extractor,
        "confidence": confidence,
    }


def stable_id(prefix: str, *parts: str) -> str:
    raw = "||".join(parts)
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


if __name__ == "__main__":
    main()
