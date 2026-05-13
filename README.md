# Fish/Aquaculture Literature Local RAG Demo

This is a small, reproducible local RAG demo for fishery and aquaculture PDF literature.

The current pipeline is:

```text
PDF loading -> text cleaning -> chunk splitting -> embedding -> ChromaDB -> hybrid retrieval -> DeepSeek answer -> cited sources
```

This repository is a starting point, not the final knowledge graph system. The longer-term target is recorded in [PROJECT_GOAL.md](PROJECT_GOAL.md).

## What Is Included

- PDF ingestion and chunking: `populate_database.py`
- Embedding provider setup: `get_embedding_function.py`
- Retrieval and QA: `query_data.py`
- Streamlit demo UI: `app.py`
- Config loader: `rag_config.py`
- Utility scripts: `scripts/`
- Minimal tests: `test_rag.py`
- Project roadmap: `PROJECT_GOAL.md`

## What Is Not Included

The following local artifacts are intentionally not committed:

- `.env`
- PDF papers under `data/papers/`
- Chroma vector database under `data/vector_db/`
- Local embedding model weights under `models/`
- Python caches such as `__pycache__/` and `.pytest_cache/`

This keeps the GitHub version small, clean, and safe to publish.

## Setup

Create or activate the Python environment first. The original local environment used:

```bash
conda activate rag
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local `.env` file from `.env.example`:

```text
DEEPSEEK_API_KEY=your_deepseek_api_key
```

`.env` is ignored by git and must not be committed.

## Prepare Local Data

Put fish/aquaculture PDF papers into:

```text
data/papers/
```

Download or verify the local embedding model:

```bash
python scripts/download_embedding_model.py
```

Build the Chroma vector database:

```bash
python scripts/rebuild_index.py
```

Inspect the vector database:

```bash
python scripts/inspect_vector_db.py
```

## Query From CLI

```bash
python query_data.py "cyp19a 在鱼类温度依赖性性别分化中和 DNA 甲基化有什么关系？"
```

The response is JSON with:

- `answer`
- `cited_sources`

`cited_sources` includes fields such as:

- `pdf_file`
- `page_number`
- `chunk_id`
- `paper_id`
- `retrieval_method`
- `score`
- `matched_terms`
- `snippet`

## Streamlit Demo

Start the local UI:

```bash
streamlit run app.py
```

The UI provides:

- literature QA
- retrieval-only chunk inspection
- knowledge base status
- PDF list display
- API check button
- vector database rebuild button

Streamlit is only the demo interface. The long-term architecture will likely use FastAPI, a real frontend, a vector database, keyword search, and Neo4j.

## Useful Commands

```bash
python scripts/check_api.py
python scripts/download_embedding_model.py
python scripts/rebuild_index.py
python scripts/inspect_vector_db.py
python query_data.py "your question"
streamlit run app.py
pytest
```

## Tests

```bash
pytest
```

The current tests cover basic text cleaning, chunk metadata, query expansion, keyword recall, and hybrid citation formatting.
