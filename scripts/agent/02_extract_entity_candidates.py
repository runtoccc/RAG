from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterator


DEFAULT_INPUT = "data/passages/openscholar_passages.jsonl"
DEFAULT_COMPAT_INPUT = "data/passages/scientific_passages.jsonl"
DEFAULT_OUTPUT = "data/agent/entity_candidates.jsonl"

SPECIES_RE = re.compile(r"\b([A-Z][a-z]+) ([a-z]{3,})(?:\b|(?=[,.;:)]))")
GENE_RE = re.compile(r"\b([a-z]{2,6}\d{0,3}[a-z]{0,3}\d{0,3}(?:[.-]?\d+[a-z]?)?)\b")
METHOD_RE = re.compile(
    r"\b(RNA-seq|WGBS|qPCR|RT-qPCR|ELISA|ChIP-seq|bisulfite sequencing|whole-genome bisulfite sequencing)\b",
    re.I,
)

TERM_TYPES = {
    "chemical": [
        "bisphenol a",
        "cadmium",
        "microplastic",
        "microplastics",
        "copper",
        "mercury",
        "lead",
        "arsenic",
        "xenobiotic",
    ],
    "phenotype": [
        "sex reversal",
        "temperature-dependent sex determination",
        "growth",
        "stress response",
        "immune response",
        "reproduction",
        "methylation",
        "dna methylation",
        "epigenetic",
    ],
    "tissue": [
        "gonad",
        "testis",
        "ovary",
        "liver",
        "brain",
        "muscle",
        "hypothalamus",
        "gill",
        "intestine",
    ],
    "environmental_factor": [
        "temperature",
        "hypoxia",
        "salinity",
        "acidification",
        "pollution",
        "thermal stress",
    ],
    "disease": [
        "infection",
        "disease",
        "vibriosis",
        "streptococcosis",
        "parasite",
    ],
    "protein": [
        "aromatase",
        "dmrt1",
        "foxl2",
    ],
}

GENE_ALLOWLIST = {
    "cyp19a1a",
    "cyp19a1b",
    "dmrt1",
    "foxl2",
    "sox9",
    "amh",
    "gsdf",
    "esr1",
    "esr2",
    "dnmt1",
    "dnmt3",
    "tet1",
    "tet2",
    "tet3",
}

SPECIES_GENUS_ALLOWLIST = {
    "Astatotilapia",
    "Astyanax",
    "Carassius",
    "Clarias",
    "Coregonus",
    "Cynoglossus",
    "Cyprinus",
    "Danio",
    "Dicentrarchus",
    "Epinephelus",
    "Fundulus",
    "Gadus",
    "Gasterosteus",
    "Hippoglossus",
    "Ictalurus",
    "Larimichthys",
    "Lates",
    "Nothobranchius",
    "Oncorhynchus",
    "Oreochromis",
    "Oryzias",
    "Paralichthys",
    "Poecilia",
    "Pundamilia",
    "Salmo",
    "Sparus",
    "Takifugu",
    "Tetraodon",
    "Xiphophorus",
}

COMMON_SPECIES_TERMS = [
    "zebrafish",
    "nile tilapia",
    "atlantic salmon",
    "rainbow trout",
    "european sea bass",
    "sea bass",
    "medaka",
    "common carp",
    "goldfish",
    "channel catfish",
    "turbot",
    "three-spined stickleback",
    "stickleback",
    "cichlid",
    "cichlids",
]


def main() -> None:
    args = parse_args()
    input_path = resolve_input(Path(args.input))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_passages = 0
    total_candidates = 0
    seen: set[tuple[str, str, str, str]] = set()
    with output_path.open("w", encoding="utf-8") as output_file:
        for passage in iter_jsonl(input_path):
            total_passages += 1
            for candidate in extract_candidates(passage):
                key = (
                    candidate["paper_id"],
                    candidate["passage_id"],
                    candidate["entity_type"],
                    candidate["normalized_name"],
                )
                if key in seen:
                    continue
                seen.add(key)
                total_candidates += 1
                output_file.write(json.dumps(candidate, ensure_ascii=False) + "\n")

    print(f"[entity-candidates] input={input_path}")
    print(f"[entity-candidates] passages={total_passages}")
    print(f"[entity-candidates] candidates={total_candidates}")
    print(f"[entity-candidates] output={output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract first-pass fish/aquaculture entity candidates.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve_input(path: Path) -> Path:
    if path.exists():
        return path
    compat = Path(DEFAULT_COMPAT_INPUT)
    if path == Path(DEFAULT_INPUT) and compat.exists():
        return compat
    raise FileNotFoundError(path)


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_candidates(passage: dict) -> Iterator[dict]:
    text = passage.get("text") or ""
    lowered = text.lower()
    yield from extract_species(passage, text)
    yield from extract_gene_allowlist(passage, text)
    yield from extract_methods(passage, text)
    for entity_type, terms in TERM_TYPES.items():
        for term in terms:
            start = lowered.find(term)
            if start == -1:
                continue
            end = start + len(term)
            yield make_candidate(
                passage,
                surface=text[start:end],
                normalized_name=term,
                entity_type=entity_type,
                evidence_text=evidence_window(text, start, end),
                extractor="term_dictionary_v1",
                confidence=0.72,
            )


def extract_species(passage: dict, text: str) -> Iterator[dict]:
    lowered = text.lower()
    for term in COMMON_SPECIES_TERMS:
        start = lowered.find(term)
        if start == -1:
            continue
        end = start + len(term)
        yield make_candidate(
            passage,
            surface=text[start:end],
            normalized_name=term,
            entity_type="species",
            evidence_text=evidence_window(text, start, end),
            extractor="species_dictionary_v1",
            confidence=0.8,
        )

    for match in SPECIES_RE.finditer(text):
        genus = match.group(1)
        species = match.group(2)
        if genus not in SPECIES_GENUS_ALLOWLIST:
            continue
        surface = f"{genus} {species}"
        yield make_candidate(
            passage,
            surface=surface,
            normalized_name=surface.lower(),
            entity_type="species",
            evidence_text=evidence_window(text, match.start(1), match.end(1)),
            extractor="latin_binomial_regex_v1",
            confidence=0.74,
        )


def extract_gene_allowlist(passage: dict, text: str) -> Iterator[dict]:
    lowered = text.lower()
    for gene in GENE_ALLOWLIST:
        for match in re.finditer(rf"\b{re.escape(gene)}\b", lowered):
            yield make_candidate(
                passage,
                surface=text[match.start() : match.end()],
                normalized_name=gene,
                entity_type="gene",
                evidence_text=evidence_window(text, match.start(), match.end()),
                extractor="gene_allowlist_v1",
                confidence=0.82,
            )


def extract_methods(passage: dict, text: str) -> Iterator[dict]:
    for match in METHOD_RE.finditer(text):
        surface = match.group(1)
        yield make_candidate(
            passage,
            surface=surface,
            normalized_name=surface.lower(),
            entity_type="method",
            evidence_text=evidence_window(text, match.start(1), match.end(1)),
            extractor="method_regex_v1",
            confidence=0.78,
        )


def make_candidate(
    passage: dict,
    surface: str,
    normalized_name: str,
    entity_type: str,
    evidence_text: str,
    extractor: str,
    confidence: float,
) -> dict:
    paper_id = passage.get("paper_id") or ""
    passage_id = passage.get("passage_id") or ""
    entity_id = stable_id("ent", entity_type, normalized_name)
    mention_id = stable_id("ment", paper_id, passage_id, entity_type, normalized_name)
    return {
        "entity_id": entity_id,
        "mention_id": mention_id,
        "surface": surface,
        "normalized_name": normalized_name,
        "entity_type": entity_type,
        "paper_id": paper_id,
        "passage_id": passage_id,
        "source_file": passage.get("source_file"),
        "doi": passage.get("doi"),
        "evidence_text": evidence_text,
        "extractor": extractor,
        "confidence": confidence,
    }


def evidence_window(text: str, start: int, end: int, radius: int = 220) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return " ".join(text[left:right].split())


def stable_id(prefix: str, *parts: str) -> str:
    raw = "||".join(parts)
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


if __name__ == "__main__":
    main()
