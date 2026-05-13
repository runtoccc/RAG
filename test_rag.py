from langchain_core.documents import Document

from populate_database import calculate_chunk_ids, clean_text, make_paper_id, split_documents
from query_data import (
    RetrievalCandidate,
    expand_query,
    format_cited_sources,
    format_hybrid_cited_sources,
    keyword_search,
)


def test_clean_text_removes_pdf_line_noise():
    raw_text = "DNA methyla-\n tion   changes\n\n\nin Nile tilapia"
    assert clean_text(raw_text) == "DNA methylation changes\n\nin Nile tilapia"


def test_split_documents_adds_required_fish_literature_metadata():
    documents = [
        Document(
            page_content=(
                "Fish growth and aquaculture epigenetics are discussed in this paper. "
                * 20
            ),
            metadata={
                "source": "data/papers/Nile tilapia growth.pdf",
                "source_file": "Nile tilapia growth.pdf",
                "page_number": 3,
                "paper_id": make_paper_id("Nile tilapia growth.pdf"),
            },
        )
    ]
    config = {"chunking": {"chunk_size": 120, "chunk_overlap": 20}}

    chunks = split_documents(documents, config)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["source_file"] == "Nile tilapia growth.pdf"
        assert chunk.metadata["page_number"] == 3
        assert chunk.metadata["paper_id"] == "nile-tilapia-growth"
        assert chunk.metadata["chunk_id"].startswith("nile-tilapia-growth:p3:c")


def test_calculate_chunk_ids_resets_for_each_page():
    chunks = calculate_chunk_ids(
        [
            Document(
                page_content="a",
                metadata={
                    "source_file": "paper.pdf",
                    "page_number": 1,
                    "paper_id": "paper",
                },
            ),
            Document(
                page_content="b",
                metadata={
                    "source_file": "paper.pdf",
                    "page_number": 1,
                    "paper_id": "paper",
                },
            ),
            Document(
                page_content="c",
                metadata={
                    "source_file": "paper.pdf",
                    "page_number": 2,
                    "paper_id": "paper",
                },
            ),
        ]
    )

    assert [chunk.metadata["chunk_id"] for chunk in chunks] == [
        "paper:p1:c0",
        "paper:p1:c1",
        "paper:p2:c0",
    ]


def test_format_cited_sources_returns_pdf_page_chunk_and_snippet():
    document = Document(
        page_content=" ".join(["zebrafish methylation evidence"] * 80),
        metadata={
            "source_file": "zebrafish.pdf",
            "page_number": 5,
            "chunk_id": "zebrafish:p5:c2",
            "paper_id": "zebrafish",
        },
    )
    sources = format_cited_sources([(document, 0.123)])

    assert sources[0]["pdf_file"] == "zebrafish.pdf"
    assert sources[0]["page_number"] == 5
    assert sources[0]["chunk_id"] == "zebrafish:p5:c2"
    assert len(sources[0]["snippet"]) <= 700


def test_expand_query_adds_dnmt_methylation_terms():
    queries = expand_query("鱼类中 dnmt1、dnmt3a、dnmt3b 分别和哪些甲基化过程相关？")
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
    assert sources[0]["matched_terms"] == ["dnmt1", "maintenance"]
