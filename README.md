# Local autoSKG

This project is now a local autoSKG-style backend for building a scientific knowledge graph from PDFs.

It mirrors the original Docker image layout:

```text
preprocess.py
-> graphrag index
-> postprocess.py
```

Local adaptations:

- PDFs live in `data/papers/`.
- GROBID is expected at `http://localhost:8070`.
- DeepSeek is used through its OpenAI-compatible API.
- GraphRAG project output lives in `data/autoskg/kg_project/`.
- Prompts and settings are copied from `autoskg_fs/app/kg_project`.

## Install

Use the `rag` conda environment:

```bash
conda run -n rag python -m pip install -r requirements.txt
```

## Run

Full pipeline:

```bash
conda run -n rag python entrypoint.py
```

Partial runs:

```bash
conda run -n rag python preprocess.py
conda run -n rag python entrypoint.py --skip-preprocess
conda run -n rag python postprocess.py
```

Strict CC-BY mode, matching the original autoSKG license gate:

```bash
conda run -n rag python entrypoint.py --license-policy cc_by
```

Default mode is `allow_all`, because these are local PDFs.

## Outputs

```text
data/autoskg/kg_project/input/
data/autoskg/kg_project/cache/
data/autoskg/kg_project/logs/
data/autoskg/kg_project/output/
```

Key final files after GraphRAG succeeds:

```text
create_final_entities.parquet
create_final_relationships.parquet
create_final_text_units.parquet
create_final_documents.parquet
metadata.parquet
```
