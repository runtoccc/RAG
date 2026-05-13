import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

from langchain_core.documents import Document
from openai import OpenAI

from get_embedding_function import get_embedding_function
from env_loader import load_dotenv
from rag_config import load_config, project_path


PROMPT_TEMPLATE = """
You are a fishery and aquaculture literature assistant.
Answer the question using only the following retrieved PDF excerpts.
First judge whether the excerpts contain enough direct evidence.
If the excerpts do not contain enough evidence, say that the current corpus does not provide enough information and cite the closest evidence.
Answer in Chinese unless the user explicitly asks for another language.
Keep the answer concise, but do not be overly conservative when English evidence supports a careful Chinese synthesis.
Separate direct evidence from inferred conclusions when the conclusion is a synthesis.
Always cite source markers like [S1], [S2] and mention chunk_id for key evidence.
For DNMT questions, if excerpts mention maintenance dnmt1 and de novo methyltransferases dnmt3,
you may infer that dnmt1 relates to maintenance DNA methylation and dnmt3a/dnmt3b relate to de novo DNA methylation.

{context}

---

Question: {question}
"""


GENE_OR_ABBREVIATION_PATTERN = re.compile(
    r"\b(?:[A-Za-z]{2,}\d+[A-Za-z]?|[A-Z]{2,}[A-Z0-9-]*|[A-Za-z]{2,}[A-Z][A-Za-z0-9-]*)\b",
)
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}")
STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "how",
    "into",
    "the",
    "their",
    "them",
    "these",
    "this",
    "those",
    "what",
    "when",
    "where",
    "which",
    "with",
}


@dataclass
class RetrievalCandidate:
    doc: Document
    chunk_id: str
    vector_score: float = 0.0
    bm25_score: float = 0.0
    keyword_score: float = 0.0
    score: float = 0.0
    vector_distance: float | None = None
    methods: set[str] = field(default_factory=set)
    matched_terms: set[str] = field(default_factory=set)


def main():
    configure_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--k", type=int, default=None, help="Number of chunks to retrieve.")
    args = parser.parse_args()
    try:
        result = query_rag(args.query_text, top_k=args.k)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as error:
        print(
            json.dumps(
                {"error": str(error), "type": error.__class__.__name__},
                ensure_ascii=False,
                indent=2,
            )
        )
        sys.exit(1)


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def query_rag(
    question: str, top_k: int | None = None, k: int | None = None
) -> dict[str, Any]:
    load_dotenv()
    config = load_config()
    effective_top_k = top_k or k or int(config["retrieval"]["top_k"])
    cited_sources = retrieve_chunks(question, top_k=effective_top_k, config=config)
    prompt = build_prompt(question, cited_sources)

    model = build_llm(config)
    response_text = model.invoke(prompt)

    return {
        "answer": response_text,
        "cited_sources": cited_sources,
    }


def retrieve_chunks(
    question: str, top_k: int = 5, config: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    config = config or load_config()
    chroma_path = project_path(config["paths"]["chroma_dir"])
    validate_chroma_path(chroma_path)

    embedding_function = get_embedding_function(config)
    Chroma = get_chroma_class()
    db = Chroma(persist_directory=str(chroma_path), embedding_function=embedding_function)

    existing_items = db.get(include=[])
    if not existing_items.get("ids"):
        raise RuntimeError(
            "Chroma vector database is empty. Run: python scripts/rebuild_index.py"
        )

    expanded_queries = expand_query(question)
    all_candidates: dict[str, RetrievalCandidate] = {}
    vector_k = max(top_k * 3, top_k, 8)

    for query in expanded_queries:
        vector_results = db.similarity_search_with_score(query, k=vector_k)
        merge_vector_results(all_candidates, vector_results)

    all_docs = load_all_chroma_documents(db)
    keyword_terms = extract_keyword_terms(question, expanded_queries)
    bm25_results = bm25_search(all_docs, keyword_terms, limit=max(top_k * 4, 20))
    keyword_results = keyword_search(all_docs, keyword_terms, limit=max(top_k * 4, 20))
    merge_scored_results(all_candidates, bm25_results, method="bm25")
    merge_scored_results(all_candidates, keyword_results, method="keyword")

    ranked_candidates = rank_candidates(all_candidates.values())
    return format_hybrid_cited_sources(ranked_candidates[:top_k])


def validate_chroma_path(chroma_path: Path) -> None:
    if not chroma_path.exists():
        raise RuntimeError(
            "Chroma vector database does not exist. Run: python scripts/rebuild_index.py"
        )


def build_prompt(query_text: str, cited_sources: list[dict[str, Any]]) -> str:
    context_text = "\n\n---\n\n".join(
        [
            f"[S{index}] "
            f"{source['pdf_file']} p.{source['page_number']} "
            f"chunk {source['chunk_id']} "
            f"method={source.get('retrieval_method', 'vector')} "
            f"score={source.get('score', 0):.4f}\n"
            f"{source['snippet']}"
            for index, source in enumerate(cited_sources, start=1)
        ]
    )
    return PROMPT_TEMPLATE.format(context=context_text, question=query_text)


def get_chroma_class():
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain.vectorstores.chroma import Chroma

    return Chroma


def build_llm(config: dict[str, Any]):
    load_dotenv()
    llm_config = config["llm"]
    provider = llm_config.get("provider", "deepseek").lower()

    if provider in {"deepseek", "openai_compatible", "openai"}:
        api_key_env = llm_config.get("api_key_env", "DEEPSEEK_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing LLM API key. Set environment variable {api_key_env}.")

        return OpenAICompatibleChatClient(
            model=llm_config["model"],
            api_key=api_key,
            base_url=llm_config.get("base_url"),
            temperature=float(llm_config.get("temperature", 0.2)),
            max_tokens=int(llm_config.get("max_tokens", 2048)),
        )

    raise ValueError(
        f"Unsupported LLM provider '{provider}'. "
        "Use 'deepseek', 'openai_compatible', or 'openai'."
    )


class OpenAICompatibleChatClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def invoke(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""


def format_cited_sources(results) -> list[dict[str, Any]]:
    cited_sources = []

    for doc, score in results:
        metadata = doc.metadata
        cited_sources.append(
            {
                "pdf_file": metadata.get("source_file"),
                "page_number": metadata.get("page_number"),
                "chunk_id": metadata.get("chunk_id") or metadata.get("id"),
                "paper_id": metadata.get("paper_id"),
                "title": metadata.get("title"),
                "doi": metadata.get("doi") or metadata.get("DOI"),
                "year": metadata.get("year"),
                "journal": metadata.get("journal"),
                "section": metadata.get("section"),
                "retrieval_method": metadata.get("retrieval_method", "vector"),
                "score": score,
                "snippet": make_snippet(doc.page_content),
            }
        )

    return cited_sources


def expand_query(question: str) -> list[str]:
    queries = [question.strip()]
    lower_question = question.lower()
    terms = extract_keyword_terms(question, [question])
    has_dnmt_intent = (
        any(term.startswith("dnmt") for term in terms)
        or "methyltransferase" in lower_question
        or "甲基转移酶" in question
    )

    if terms:
        queries.append(" ".join(terms))

    if "甲基化" in question or "methyl" in lower_question:
        queries.extend(
            [
                "DNA methylation fish",
                "epigenetic regulation fish",
                "methylation epigenetic aquaculture",
            ]
        )

    if has_dnmt_intent:
        queries.extend(
            [
                "DNA methyltransferase dnmt fish",
                "dnmt1 dnmt3a dnmt3b DNA methylation fish",
                "maintenance DNA methylation dnmt1",
                "de novo DNA methylation dnmt3a dnmt3b",
                "de novo methyltransferases dnmt3",
            ]
        )

    if "cyp19a" in lower_question or "aromatase" in lower_question or "芳香化酶" in question:
        queries.extend(
            [
                "cyp19a aromatase promoter methylation fish",
                "cyp19a temperature sex differentiation DNA methylation",
                "gonadal aromatase cyp19a promoter methylation",
            ]
        )

    if any(term in question for term in ["污染物", "外源", "暴露", "跨代"]) or any(
        term in lower_question for term in ["xenobiotic", "exposure", "pollutant", "transgenerational"]
    ):
        queries.extend(
            [
                "xenobiotic exposure zebrafish transgenerational epigenetic",
                "pollutant exposure DNA methylation zebrafish",
                "transgenerational DNA methylation fish exposure",
            ]
        )

    if any(term in question for term in ["肌肉", "生长", "早期发育"]) or any(
        term in lower_question for term in ["muscle", "growth", "early development"]
    ):
        queries.extend(
            [
                "muscle methylome growth fish early development",
                "muscle DNA methylation growth traits fish",
                "early developmental stages muscle methylation fish",
            ]
        )

    if "鱼" in question or "水产" in question or "fish" in lower_question:
        queries.extend(["fish aquaculture", "teleost fish"])

    return unique_keep_order([query for query in queries if query])


def extract_keyword_terms(question: str, expanded_queries: list[str] | None = None) -> list[str]:
    text = " ".join([question, *(expanded_queries or [])])
    terms: list[str] = []

    for match in GENE_OR_ABBREVIATION_PATTERN.findall(text):
        normalized = match.lower().strip("-")
        if len(normalized) >= 3 and normalized not in STOPWORDS:
            terms.append(normalized)

    domain_terms = {
        "dnmt1": ["dnmt1", "maintenance", "methylation"],
        "dnmt3a": ["dnmt3a", "dnmt3", "de novo", "methylation"],
        "dnmt3b": ["dnmt3b", "dnmt3", "de novo", "methylation"],
        "cyp19a": ["cyp19a", "aromatase", "promoter", "methylation"],
        "芳香化酶": ["cyp19a", "aromatase"],
        "甲基化": ["methylation", "dna methylation"],
        "鱼": ["fish", "teleost", "aquaculture"],
        "水产": ["fish", "aquaculture"],
        "斑马鱼": ["zebrafish", "danio"],
        "温度": ["temperature"],
        "性别": ["sex", "gonad", "differentiation"],
        "污染物": ["pollutant", "xenobiotic", "exposure"],
        "外源": ["xenobiotic", "exposure"],
        "暴露": ["exposure"],
        "跨代": ["transgenerational"],
        "肌肉": ["muscle"],
        "生长": ["growth"],
        "早期发育": ["early", "development"],
    }
    for trigger, additions in domain_terms.items():
        if trigger in question.lower() or trigger in question:
            terms.extend(additions)

    for token in TOKEN_PATTERN.findall(text):
        normalized = token.lower()
        if normalized in STOPWORDS:
            continue
        if normalized in {"dna", "rna", "doi"} or len(normalized) >= 4:
            terms.append(normalized)

    return unique_keep_order(terms)


def merge_vector_results(
    candidates: dict[str, RetrievalCandidate], results: list[tuple[Document, float]]
) -> None:
    for doc, distance in results:
        candidate = get_or_create_candidate(candidates, doc)
        candidate.methods.add("vector")
        candidate.vector_distance = (
            distance
            if candidate.vector_distance is None
            else min(candidate.vector_distance, distance)
        )
        candidate.vector_score = max(candidate.vector_score, distance_to_similarity(distance))


def merge_scored_results(
    candidates: dict[str, RetrievalCandidate],
    results: list[tuple[Document, float, list[str]]],
    method: str,
) -> None:
    for doc, score, matched_terms in results:
        candidate = get_or_create_candidate(candidates, doc)
        candidate.methods.add(method)
        candidate.matched_terms.update(matched_terms)
        if method == "bm25":
            candidate.bm25_score = max(candidate.bm25_score, score)
        elif method == "keyword":
            candidate.keyword_score = max(candidate.keyword_score, score)


def get_or_create_candidate(
    candidates: dict[str, RetrievalCandidate], doc: Document
) -> RetrievalCandidate:
    chunk_id = get_chunk_id(doc)
    if chunk_id not in candidates:
        candidates[chunk_id] = RetrievalCandidate(doc=doc, chunk_id=chunk_id)
    return candidates[chunk_id]


def get_chunk_id(doc: Document) -> str:
    metadata = doc.metadata or {}
    return (
        metadata.get("chunk_id")
        or metadata.get("id")
        or f"{metadata.get('source_file', 'unknown')}:{metadata.get('page_number', 'unknown')}:{hash(doc.page_content)}"
    )


def distance_to_similarity(distance: float) -> float:
    if distance < 0:
        return 0.0
    return 1.0 / (1.0 + distance)


def load_all_chroma_documents(db) -> list[Document]:
    items = db.get(include=["documents", "metadatas"])
    documents = items.get("documents") or []
    metadatas = items.get("metadatas") or [{} for _ in documents]
    return [
        Document(page_content=document or "", metadata=metadata or {})
        for document, metadata in zip(documents, metadatas)
    ]


def keyword_search(
    documents: list[Document], terms: list[str], limit: int = 20
) -> list[tuple[Document, float, list[str]]]:
    if not terms:
        return []

    results: list[tuple[Document, float, list[str]]] = []
    for doc in documents:
        haystack = f"{doc.page_content} {json.dumps(doc.metadata, ensure_ascii=False)}".lower()
        matched = [term for term in terms if term.lower() in haystack]
        if not matched:
            continue
        exact_gene_bonus = sum(2.0 for term in matched if re.search(r"\d", term))
        phrase_bonus = sum(1.5 for term in matched if " " in term)
        score = len(set(matched)) + exact_gene_bonus + phrase_bonus
        results.append((doc, score, unique_keep_order(matched)))

    return sorted(results, key=lambda item: item[1], reverse=True)[:limit]


def bm25_search(
    documents: list[Document], terms: list[str], limit: int = 20
) -> list[tuple[Document, float, list[str]]]:
    query_terms = [term for term in terms if " " not in term]
    if not query_terms or not documents:
        return []

    tokenized_documents = [tokenize(doc.page_content) for doc in documents]
    doc_count = len(tokenized_documents)
    avg_doc_len = sum(len(tokens) for tokens in tokenized_documents) / max(doc_count, 1)
    document_frequencies = Counter()
    for tokens in tokenized_documents:
        document_frequencies.update(set(tokens))

    k1 = 1.5
    b = 0.75
    results: list[tuple[Document, float, list[str]]] = []
    for doc, tokens in zip(documents, tokenized_documents):
        if not tokens:
            continue
        term_counts = Counter(tokens)
        score = 0.0
        matched_terms: list[str] = []
        for term in query_terms:
            normalized_term = term.lower()
            term_frequency = term_counts.get(normalized_term, 0)
            if not term_frequency:
                continue
            matched_terms.append(normalized_term)
            doc_frequency = document_frequencies.get(normalized_term, 0)
            idf = math.log(1 + (doc_count - doc_frequency + 0.5) / (doc_frequency + 0.5))
            denominator = term_frequency + k1 * (
                1 - b + b * len(tokens) / max(avg_doc_len, 1)
            )
            score += idf * (term_frequency * (k1 + 1) / denominator)
        if score > 0:
            results.append((doc, score, unique_keep_order(matched_terms)))

    return sorted(results, key=lambda item: item[1], reverse=True)[:limit]


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def rank_candidates(candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
    max_bm25 = max((candidate.bm25_score for candidate in candidates), default=0.0)
    max_keyword = max((candidate.keyword_score for candidate in candidates), default=0.0)

    for candidate in candidates:
        normalized_bm25 = candidate.bm25_score / max_bm25 if max_bm25 else 0.0
        normalized_keyword = candidate.keyword_score / max_keyword if max_keyword else 0.0
        hybrid_bonus = 0.12 * max(0, len(candidate.methods) - 1)
        candidate.score = (
            0.55 * candidate.vector_score
            + 0.30 * normalized_bm25
            + 0.15 * normalized_keyword
            + hybrid_bonus
        )

    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.score,
            len(candidate.methods),
            candidate.keyword_score,
            candidate.bm25_score,
        ),
        reverse=True,
    )


def format_hybrid_cited_sources(
    candidates: list[RetrievalCandidate],
) -> list[dict[str, Any]]:
    cited_sources = []

    for candidate in candidates:
        metadata = candidate.doc.metadata
        cited_sources.append(
            {
                "pdf_file": metadata.get("source_file"),
                "page_number": metadata.get("page_number"),
                "chunk_id": metadata.get("chunk_id") or metadata.get("id"),
                "paper_id": metadata.get("paper_id"),
                "title": metadata.get("title"),
                "doi": metadata.get("doi") or metadata.get("DOI"),
                "year": metadata.get("year"),
                "journal": metadata.get("journal"),
                "section": metadata.get("section"),
                "retrieval_method": "+".join(sorted(candidate.methods)) or "vector",
                "score": candidate.score,
                "vector_score": candidate.vector_score,
                "bm25_score": candidate.bm25_score,
                "keyword_score": candidate.keyword_score,
                "matched_terms": sorted(candidate.matched_terms),
                "snippet": make_snippet(candidate.doc.page_content),
            }
        )

    return cited_sources


def unique_keep_order(values: list[str]) -> list[str]:
    seen = set()
    unique_values = []
    for value in values:
        normalized = value.strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        unique_values.append(normalized)
    return unique_values


def make_snippet(text: str, max_chars: int = 700) -> str:
    compact_text = " ".join(text.split())
    if len(compact_text) <= max_chars:
        return compact_text
    return compact_text[: max_chars - 3].rstrip() + "..."


if __name__ == "__main__":
    main()
