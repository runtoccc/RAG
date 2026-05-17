from langchain_core.documents import Document

from populate_database import (
    normalize_passage_text,
    scientific_passage_record_to_document,
)
from query_data import (
    RetrievalCandidate,
    expand_query,
    extract_display_snippet,
    format_cited_sources,
    format_hybrid_cited_sources,
    keyword_search,
)
from scripts.check_passage_quality import calculate_passage_quality_stats


def test_scientific_passage_record_becomes_chroma_document():
    record = {
        "passage_id": "local_000001::block::0000",
        "paper_id": "local_000001",
        "title": "Zebrafish methylation",
        "block_index": 0,
        "block_words": 5,
        "text": "DNA methylation evidence in zebrafish.",
        "embedding_text": "Zebrafish methylation\n\nDNA methylation evidence in zebrafish.",
        "source_file": "zebrafish.pdf",
        "page": None,
        "doi": "10.123/example",
        "chunk_style": "openscholar_256w_title_prefix",
        "section_title": None,
        "section_type": None,
    }

    document = scientific_passage_record_to_document(record)

    assert document.metadata["chunk_id"] == "local_000001::block::0000"
    assert document.metadata["paper_id"] == "local_000001"
    assert document.metadata["title"] == "Zebrafish methylation"
    assert document.metadata["section"] == ""
    assert document.metadata["section_type"] == ""
    assert document.metadata["block_index"] == 0
    assert document.metadata["block_words"] == 5
    assert document.metadata["chunk_style"] == "openscholar_256w_title_prefix"
    assert document.metadata["source_file"] == "zebrafish.pdf"
    assert document.metadata["page_number"] == "unknown"
    assert document.metadata["doi"] == "10.123/example"
    assert document.metadata["chunk_source"] == "scientific_passages_jsonl"
    assert document.page_content == "Zebrafish methylation\n\nDNA methylation evidence in zebrafish."


def test_normalize_passage_text_keeps_paragraph_breaks():
    raw_text = "DNA methylation   changes\n\n\nin Nile tilapia"
    assert normalize_passage_text(raw_text) == "DNA methylation changes\n\nin Nile tilapia"


def test_format_cited_sources_returns_pdf_page_chunk_and_snippet():
    document = Document(
        page_content=(
            "Title: zebrafish methylation\n"
            "Section: Results\n"
            "Page: 5\n"
            f"Text: {' '.join(['zebrafish methylation evidence'] * 80)}"
        ),
        metadata={
            "source_file": "zebrafish.pdf",
            "page_number": 5,
            "chunk_id": "zebrafish:p5:c2",
            "paper_id": "zebrafish",
            "title": "zebrafish methylation",
            "section": "Results",
            "is_reference_section": False,
        },
    )
    sources = format_cited_sources([(document, 0.123)])

    assert sources[0]["pdf_file"] == "zebrafish.pdf"
    assert sources[0]["source_file"] == "zebrafish.pdf"
    assert sources[0]["page_number"] == 5
    assert sources[0]["chunk_id"] == "zebrafish:p5:c2"
    assert sources[0]["title"] == "zebrafish methylation"
    assert sources[0]["section"] == "Results"
    assert sources[0]["is_reference_section"] is False
    assert "Title:" not in sources[0]["snippet"]
    assert "Section:" not in sources[0]["snippet"]
    assert "Page:" not in sources[0]["snippet"]
    assert len(sources[0]["snippet"]) <= 700


def test_extract_display_snippet_removes_scientific_passage_prefix():
    page_content = (
        "Title: annurev animal 021419 083634\n"
        "Section: Results\n"
        "Page: 14\n"
        "Text: Dnmt1 maintenance methylation evidence."
    )

    assert extract_display_snippet(page_content) == "Dnmt1 maintenance methylation evidence."


def test_extract_display_snippet_removes_openscholar_title_prefix():
    page_content = "paper title\n\nDnmt1 maintenance methylation evidence."

    assert extract_display_snippet(page_content) == "Dnmt1 maintenance methylation evidence."


def test_expand_query_adds_dnmt_methylation_terms():
    queries = expand_query("fish dnmt1 dnmt3a dnmt3b methylation")
    expanded_text = " ".join(queries).lower()

    assert "maintenance dna methylation dnmt1" in expanded_text
    assert "de novo dna methylation dnmt3a dnmt3b" in expanded_text
    assert "de novo methyltransferases dnmt3" in expanded_text


def test_keyword_search_recalls_exact_gene_abbreviations():
    documents = [
        Document(
            page_content=(
                "Dnmt1 is linked to maintenance DNA methylation. "
                "Dnmt3a and Dnmt3b are de novo methyltransferases."
            ),
            metadata={
                "source_file": "annurev.pdf",
                "page_number": 14,
                "chunk_id": "annurev:p14:c4",
                "paper_id": "annurev",
            },
        ),
        Document(
            page_content="Zebrafish growth and aquaculture nutrition.",
            metadata={"chunk_id": "other:p1:c0"},
        ),
    ]

    results = keyword_search(
        documents,
        ["dnmt1", "dnmt3a", "dnmt3b", "maintenance", "de novo"],
        limit=5,
    )

    assert results[0][0].metadata["chunk_id"] == "annurev:p14:c4"
    assert {"dnmt1", "dnmt3a", "dnmt3b"}.issubset(set(results[0][2]))


def test_format_hybrid_cited_sources_includes_retrieval_details():
    document = Document(
        page_content="Dnmt1 maintenance methylation evidence.",
        metadata={
            "source_file": "annurev.pdf",
            "page_number": 14,
            "chunk_id": "annurev:p14:c4",
            "paper_id": "annurev",
            "section": "Epigenetics",
            "title": "annurev animal 021419 083634",
            "is_reference_section": False,
        },
    )
    candidate = RetrievalCandidate(
        doc=document,
        chunk_id="annurev:p14:c4",
        vector_score=0.7,
        bm25_score=2.0,
        keyword_score=5.0,
        score=0.91,
        methods={"vector", "bm25", "keyword"},
        matched_terms={"dnmt1", "maintenance"},
    )

    sources = format_hybrid_cited_sources([candidate])

    assert sources[0]["retrieval_method"] == "bm25+keyword+vector"
    assert sources[0]["score"] == 0.91
    assert sources[0]["section"] == "Epigenetics"
    assert sources[0]["title"] == "annurev animal 021419 083634"
    assert sources[0]["is_reference_section"] is False
    assert sources[0]["matched_terms"] == ["dnmt1", "maintenance"]


def test_passage_quality_stats_detects_openscholar_format_errors():
    passages = [
        {
            "passage_id": "paper-a::block::0000",
            "paper_id": "paper-a",
            "title": "paper a",
            "block_words": 256,
            "text": "fish methylation evidence",
            "embedding_text": "paper a\n\nfish methylation evidence",
            "chunk_style": "openscholar_256w_title_prefix",
            "source_file": "paper-a.pdf",
        },
        {
            "passage_id": "paper-a::block::0001",
            "paper_id": "paper-a",
            "title": "paper a",
            "block_words": 70,
            "text": "short final block",
            "embedding_text": "paper a\n\nshort final block",
            "chunk_style": "openscholar_256w_title_prefix",
            "source_file": "paper-a.pdf",
        },
        {
            "passage_id": "paper-b::section::0000",
            "paper_id": "paper-b",
            "title": "",
            "block_words": 20,
            "text": "bad block",
            "embedding_text": "Section: bad\n\nbad block",
            "chunk_style": "section_aware",
            "source_file": "",
        },
    ]

    stats = calculate_passage_quality_stats(passages)

    assert stats["total_passages"] == 3
    assert stats["count_blocks_eq_256"] == 1
    assert stats["count_blocks_lt_64"] == 1
    assert stats["has_section_in_embedding_text_count"] == 1
    assert stats["title_prefix_valid_count"] == 2
    assert stats["chunk_style_distribution"]["openscholar_256w_title_prefix"] == 2
    assert stats["bad_passage_id_count"] == 1
    assert stats["bad_chunk_style_count"] == 1
    assert stats["missing_title_count"] == 1
    assert stats["missing_source_file_count"] == 1
