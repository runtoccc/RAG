# Aquaculture Literature KG-RAG Demo

This repository is a local fish/aquaculture literature RAG prototype. It now has two layers:

1. A PDF-to-passage foundation pipeline aligned with S2ORC / peS2o / OpenScholar principles.
2. A lightweight KG-RAG agent layer that builds a rule-extracted literature knowledge graph and a clickable evidence viewer.

The project is not an official reproduction of S2ORC, peS2o, OpenScholar, or PlantScience.ai. It is a local implementation that follows their processing principles where practical.

## Current Structure

```text
PDF
-> GROBID TEI
-> local S2ORC-compatible JSON
-> metadata enrichment
-> peS2o-style clean/filter
-> OpenScholar-style 256-word passages
-> Chroma embedding
-> local KG agent
-> KG viewer / evidence answer skeleton
```

## Foundation Pipeline

The foundation lives under `scripts/pipeline/`:

- `01_grobid_pdf_to_tei.py`: PDF -> GROBID TEI XML.
- `02_tei_to_local_s2orc.py`: TEI -> local S2ORC-compatible paper object.
- `03_enrich_metadata.py`: local metadata enrichment and override support.
- `04_pes2o_filter_clean.py`: peS2o-style strict clean/filter.
- `05_openscholar_passages.py`: OpenScholar-style title-prefixed 256-word passages.
- `06_build_chroma.py`: build Chroma from final passages.

Quality checks live under `scripts/qa/`.

Run the foundation pipeline:

```bash
python scripts/run_pipeline.py --no-strict
```

Rebuild Chroma only when explicitly needed:

```bash
python scripts/run_pipeline.py --no-strict --rebuild-chroma
```

## Agent / KG Layer

The agent layer lives under `scripts/agent/`:

- `01_register_papers.py`: paper-level provenance registry.
- `02_extract_entity_candidates.py`: rule-based entity candidate extraction.
- `03_extract_relation_candidates.py`: rule-based relation candidate extraction.
- `04_build_local_kg.py`: aggregate candidate entities/relations into local KG files.
- `05_hybrid_retrieve.py`: vector + KG evidence retrieval.
- `06_answer_with_evidence.py`: evidence-grounded answer skeleton.
- `07_export_kg_viewer_data.py`: export browser-friendly KG viewer JSON.
- `08_make_kg_viewer.py`: generate a single-file interactive HTML graph.
- `09_kg_quality_report.py`: diagnose KG quality and weak spots.

Run the KG agent and generate the viewer:

```bash
python scripts/agent/run_agent_pipeline.py "How does temperature affect sex determination in fish?" --skip-vector --make-viewer
```

Or run the viewer steps manually:

```bash
python scripts/agent/07_export_kg_viewer_data.py --bundle data/agent/evidence_bundle.json
python scripts/agent/08_make_kg_viewer.py
python scripts/agent/09_kg_quality_report.py
```

Open:

```text
outputs/agent/kg_viewer.html
```

The KG edges are rule-extracted candidate relations. They are evidence pointers, not manually verified biological facts.

## Classic RAG UI

The original Streamlit UI is still available:

```bash
streamlit run app.py
```

The optional KG viewer wrapper is:

```bash
streamlit run app_kg_viewer.py
```

## Local Data

These artifacts are intentionally ignored by git:

- `.env`
- PDFs under `data/papers/`
- parsed TEI / structured / clean / passage data
- `data/agent/`
- Chroma under `data/vector_db/`
- local model weights under `models/`
- reports and generated HTML under `outputs/`
- Python caches

Keep source code and small templates in git; keep generated data local.

## Useful Commands

```bash
python scripts/download_embedding_model.py
python scripts/rebuild_index.py
python scripts/inspect_vector_db.py
python query_data.py "your question"
python scripts/check_api.py
pytest
```

## Notes

- `scripts/legacy/` contains older scripts kept for reference only.
- The active PDF-to-chunk pipeline is `scripts/pipeline/`.
- The active KG agent layer is `scripts/agent/`.
- The old top-level `kg/` placeholder was removed; KG outputs now live in `data/agent/`.
