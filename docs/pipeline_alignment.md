# Local PDF RAG Pipeline Alignment

This project is not an official reproduction of S2ORC, peS2o, or OpenScholar. It is a local implementation aligned with their principles for a fish/aquaculture literature RAG demo.

## Directory Layout

Formal preprocessing pipeline:

```text
scripts/pipeline/01_grobid_pdf_to_tei.py
scripts/pipeline/02_tei_to_local_s2orc.py
scripts/pipeline/03_enrich_metadata.py
scripts/pipeline/04_pes2o_filter_clean.py
scripts/pipeline/05_openscholar_passages.py
scripts/pipeline/06_build_chroma.py
```

Quality checks:

```text
scripts/qa/check_s2orc_quality.py
scripts/qa/check_pes2o_quality.py
scripts/qa/check_passage_quality.py
scripts/qa/pipeline_review.py
```

Legacy scripts were moved to:

```text
scripts/legacy/
```

## local S2ORC-like Structured Object

`scripts/pipeline/02_tei_to_local_s2orc.py` converts GROBID TEI XML into a local structured paper object.

It keeps:

- `title`
- `abstract`
- `metadata`
- `abstract_paragraphs`
- `body_text`
- `sections`
- `bib_entries`
- `ref_entries`

This is not official S2ORC. It does not claim Semantic Scholar resolved metadata, ScienceParse output, publisher-canonical metadata, resolved citation-to-bibliography linking, or complete figure/table/equation spans.

The canonical output names are:

```text
data/structured/local_s2orc.jsonl
data/structured/local_s2orc_enriched.jsonl
```

Compatibility outputs may still exist:

```text
data/structured/s2orc_like.jsonl
data/structured/s2orc_like_repaired.jsonl
```

These mean local S2ORC-like objects, not official S2ORC.

## Metadata Enrichment

`scripts/pipeline/03_enrich_metadata.py` applies local metadata enrichment before filtering.

It may use:

- `data/metadata/metadata_override.csv`
- Abstract/Summary/Author Summary section recovery
- filename fallback title, marked as requiring external verification

It does not make records index-ready. `index_ready` is only computed by `04_pes2o_filter_clean.py`.

## peS2o-style Filtering

`scripts/pipeline/04_pes2o_filter_clean.py` applies local peS2o-style filtering. By default it runs in practical local mode and records missing optional resources as quality flags. Strict mode is opt-in with `--strict`.

Optional strict mode requires:

- title exists
- abstract exists
- abstract has 50-1000 words
- abstract language is English
- abstract average unigram log probability is above threshold
- clean `main_text` exists
- total text has at least 500 whitespace words
- main text has at least 5 paragraphs
- year exists and is at least 1970
- document language is English
- most frequent alpha word ratio is below 7.5%
- no severe OCR spacing noise
- no bad title, bad fallback title, or unverified fallback title
- no high-confidence residue leakage

Strict mode is not the default because this repository does not ship the large unigram resource. To enable strict mode, install/provide:

```text
pycld3
data/resources/unigram_freq.csv
```

Canonical outputs:

```text
data/clean/pes2o_style.jsonl
data/clean/pes2o_style_pass.jsonl
data/clean/pes2o_style_failed.jsonl
```

Compatibility outputs:

```text
data/clean/pes2o_like.jsonl
data/clean/pes2o_like_pass.jsonl
data/clean/pes2o_like_failed.jsonl
```

These are peS2o-style local filtering outputs, not the official peS2o dataset.

## OpenScholar-style Passages

`scripts/pipeline/05_openscholar_passages.py` builds local OpenScholar-style passages.

Rules:

- input must be pass/index-ready records
- source text is `main_text`
- whitespace split
- each block has at most 256 words
- no overlap
- no section labels
- `embedding_text = title + "\n\n" + block`

Canonical output:

```text
data/passages/openscholar_passages.jsonl
```

Compatibility output:

```text
data/passages/scientific_passages.jsonl
```

These are local OpenScholar-style passages, not OpenScholar's official retrieval corpus.

## Chroma

`scripts/pipeline/06_build_chroma.py` only indexes validated passages. It refuses passages where:

- `index_ready` is not true
- `embedding_text` is not exactly `title + "\n\n" + text`

Chroma is not rebuilt unless explicitly requested:

```powershell
python scripts/run_pipeline.py --rebuild-chroma
```

## Recommended Commands

Run preprocessing without rebuilding Chroma:

```powershell
python scripts/run_pipeline.py
```

Run preprocessing without GROBID if TEI already exists:

```powershell
python scripts/run_pipeline.py --skip-grobid
```

Run preprocessing and rebuild Chroma:

```powershell
python scripts/run_pipeline.py --rebuild-chroma
```

Run strict preprocessing only after optional strict resources are available:

```powershell
python scripts/run_pipeline.py --strict
```

## 100K Scale Notes

Before scaling to 100K papers, the remaining work is:

- stronger metadata enrichment coverage
- optional strict language detection dependency management
- optional local unigram frequency resource management
- streaming readers instead of list materialization
- batch/vector collection versioning
- deduplication and long-term manifests
