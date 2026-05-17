from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI
import pandas as pd

try:
    from .common import autoskg_config, ensure_deepseek_env, load_project_config, resolve_project_path
except ImportError:
    from common import autoskg_config, ensure_deepseek_env, load_project_config, resolve_project_path


ENTITY_TYPES = (
    "Taxon, Fish Species, Aquatic Organism, Anatomical Entity, Cellular Component, "
    "Gene, Transcript, Protein, Regulatory Element, Genetic Variant, Chemical Entity, "
    "Molecular Function, Pathway, Biological Process, Phenotype, Disease/Disorder, "
    "Stress Condition, Ecological Interaction, Environmental Factor, Habitat/Ecosystem, "
    "Aquaculture Practice, Resource, Experiment/Study, Publication, Data/Measurement, "
    "Molecular Complex, Tool/Instrument"
)


def main() -> None:
    args = parse_args()
    config = autoskg_config(load_project_config())
    ensure_deepseek_env(config)
    kg_root = resolve_project_path(args.kg_root or config["root_dir"])
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("DEEPSEEK_BASE_URL"))

    output_dir = kg_root / "output"
    add_md5_to_files(output_dir)
    process_entities_file(output_dir, client, config["llm_model"], fill_missing=not args.no_fill_missing)
    process_relationships_file(output_dir)
    if args.export_jsonl:
        export_jsonl(output_dir)
    print("[autoskg-postprocess] done")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="autoSKG-style GraphRAG postprocessing.")
    parser.add_argument("--kg-root", default=None)
    parser.add_argument("--no-fill-missing", action="store_true")
    parser.add_argument("--export-jsonl", action="store_true", default=True)
    return parser.parse_args()


def add_md5_to_files(output_dir: Path) -> None:
    text_units_path = output_dir / "create_final_text_units.parquet"
    documents_path = output_dir / "create_final_documents.parquet"
    metadata_path = output_dir / "metadata.parquet"
    if not text_units_path.exists() or not documents_path.exists() or not metadata_path.exists():
        print("[autoskg-postprocess] skip md5 join: missing text_units/documents/metadata")
        return

    final_text_units = pd.read_parquet(text_units_path).rename(columns={"id": "text_unit_id"})
    final_documents = pd.read_parquet(documents_path).rename(columns={"id": "document_id"})
    metadata = pd.read_parquet(metadata_path)
    if "title" not in final_documents.columns:
        print("[autoskg-postprocess] skip md5 join: documents missing title")
        return

    final_documents["file_name"] = final_documents["title"].astype(str).str.replace(".txt", "", regex=False)
    final_documents = final_documents.merge(metadata, on="file_name", how="left")

    for kind in ["entities", "relationships"]:
        parquet_path = output_dir / f"create_final_{kind}.parquet"
        if not parquet_path.exists():
            continue
        frame = pd.read_parquet(parquet_path)
        if "text_unit_ids" not in frame.columns:
            continue
        exploded = frame.explode("text_unit_ids").rename(columns={"text_unit_ids": "text_unit_id"})
        with_docs = exploded.merge(final_text_units[["text_unit_id", "document_ids"]], on="text_unit_id", how="left")
        with_docs = with_docs.explode("document_ids").rename(columns={"document_ids": "document_id"})
        with_docs = with_docs.merge(final_documents[["document_id", "md5"]], on="document_id", how="left")
        grouped = with_docs.groupby("id").agg({"md5": lambda values: list(values.dropna().unique())}).reset_index()
        merged = frame.merge(grouped, on="id", how="left").rename(columns={"md5": "md5s"})
        merged.to_parquet(parquet_path, index=False)


def process_entities_file(output_dir: Path, client: OpenAI, model: str, fill_missing: bool) -> None:
    entities_path = output_dir / "create_final_entities.parquet"
    if not entities_path.exists():
        raise FileNotFoundError(f"Missing GraphRAG entities file: {entities_path}")

    entities = pd.read_parquet(entities_path)
    if entities.empty:
        entities.to_parquet(entities_path, index=False)
        return

    entities["type"] = entities.get("type", pd.Series(dtype=str)).astype("string")
    entities.loc[entities["type"].str.upper() == "AMINO ACID", "type"] = "PROTEIN"
    entities.loc[entities["type"].str.upper().isin(["TRANSCRIPTION", "RNA"]), "type"] = "GENE"
    entities.loc[entities["type"].str.upper() == "GEO", "type"] = "RESOURCE"
    entities = entities[~entities["type"].str.upper().isin(["PUBLICATION", "PERSON", "ORGANIZATION"])]

    if "title" in entities.columns:
        entities = entities[~entities["title"].isin(["ZYGOATE", "RODOCOCCUS FASCIANS"])]

    entities = entities.reset_index(drop=True)
    if fill_missing:
        for index, row in entities.iterrows():
            title = str(row.get("title") or "").strip()
            if not title:
                continue
            if "description" in entities.columns and pd.isna(row.get("description")):
                entities.loc[index, "description"] = fill_blank_description(client, model, title)
            if pd.isna(row.get("type")):
                entities.loc[index, "type"] = fill_blank_type(client, model, title)

    entities.to_parquet(entities_path, index=False)


def process_relationships_file(output_dir: Path) -> None:
    relationships_path = output_dir / "create_final_relationships.parquet"
    entities_path = output_dir / "create_final_entities.parquet"
    if not relationships_path.exists():
        raise FileNotFoundError(f"Missing GraphRAG relationships file: {relationships_path}")
    if not entities_path.exists():
        raise FileNotFoundError(f"Missing GraphRAG entities file: {entities_path}")

    relationships = pd.read_parquet(relationships_path)
    entities = pd.read_parquet(entities_path)
    if relationships.empty or "title" not in entities.columns:
        relationships.to_parquet(relationships_path, index=False)
        return

    valid_titles = set(entities["title"])
    relationships = relationships[
        (relationships["source"].isin(valid_titles))
        & (relationships["target"].isin(valid_titles))
        & (relationships["source"] != relationships["target"])
    ].reset_index(drop=True)
    relationships.to_parquet(relationships_path, index=False)


def fill_blank_description(client: OpenAI, model: str, entity: str) -> str:
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are a helpful aquaculture and fish biology assistant. "
                        "Provide a brief scientific description of this entity. "
                        "No extra words.\nEntity: "
                        + entity
                    ),
                }
            ],
            max_tokens=500,
        )
        return completion.choices[0].message.content or "UNKNOWN"
    except Exception as error:
        print(f"[autoskg-postprocess] description fill failed for {entity}: {error}")
        return "UNKNOWN"


def fill_blank_type(client: OpenAI, model: str, entity: str) -> str:
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Classify this entity into exactly one of the following categories: "
                        f"{ENTITY_TYPES}.\nEntity: {entity}\nReturn only the category."
                    ),
                }
            ],
            max_tokens=100,
        )
        return completion.choices[0].message.content or "UNKNOWN"
    except Exception as error:
        print(f"[autoskg-postprocess] type fill failed for {entity}: {error}")
        return "UNKNOWN"


def export_jsonl(output_dir: Path) -> None:
    for kind in ["entities", "relationships"]:
        parquet_path = output_dir / f"create_final_{kind}.parquet"
        if not parquet_path.exists():
            continue
        frame = pd.read_parquet(parquet_path)
        jsonl_path = output_dir / f"create_final_{kind}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as file:
            for row in frame.to_dict(orient="records"):
                file.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
        print(f"[autoskg-postprocess] exported {jsonl_path}")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if pd.isna(value):
        return None
    return value


if __name__ == "__main__":
    main()
