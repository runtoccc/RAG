from __future__ import annotations

import os
import hashlib
import math
import re
from typing import Any

from openai import OpenAI

from env_loader import load_dotenv
from rag_config import load_config, project_path


class OpenAICompatibleEmbeddings:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        batch_size: int = 64,
    ):
        self.model = model
        self.batch_size = batch_size
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for index in range(0, len(texts), self.batch_size):
            batch = texts[index : index + self.batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            embeddings.extend([item.embedding for item in response.data])
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding


class HashingEmbeddings:
    def __init__(self, dimensions: int = 2048):
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self._tokens(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _tokens(self, text: str) -> list[str]:
        text = text.lower()
        words = re.findall(r"[a-z0-9][a-z0-9_-]{1,}|[\u4e00-\u9fff]", text)
        bigrams = [f"{words[index]} {words[index + 1]}" for index in range(len(words) - 1)]
        return words + bigrams


class SentenceTransformerEmbeddings:
    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        passages = [self._format_passage(text) for text in texts]
        embeddings = self.model.encode(
            passages,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(
            self._format_query(text),
            normalize_embeddings=self.normalize_embeddings,
        )
        return embedding.tolist()

    def _format_query(self, text: str) -> str:
        if "e5" in self.model_name.lower() and not text.startswith("query: "):
            return f"query: {text}"
        return text

    def _format_passage(self, text: str) -> str:
        if "e5" in self.model_name.lower() and not text.startswith("passage: "):
            return f"passage: {text}"
        return text


def get_embedding_function(config: dict[str, Any] | None = None):
    load_dotenv()
    config = config or load_config()
    embedding_config = config["embedding"]
    provider = embedding_config.get("provider", "openai_compatible").lower()

    if provider == "hashing":
        return HashingEmbeddings(
            dimensions=int(embedding_config.get("dimensions", 2048))
        )

    if provider in {"sentence_transformers", "local"}:
        local_dir = embedding_config.get("local_dir")
        local_path = project_path(local_dir) if local_dir else None
        model_name = (
            str(local_path)
            if local_path and (local_path / "config.json").exists()
            else embedding_config["model"]
        )
        cache_dir = embedding_config.get("cache_dir")
        cache_path = project_path(cache_dir) if cache_dir else None
        if cache_path:
            cache_path.mkdir(parents=True, exist_ok=True)

        return SentenceTransformerEmbeddings(
            model_name=model_name,
            cache_dir=str(cache_path) if cache_path else None,
            batch_size=int(embedding_config.get("batch_size", 64)),
            normalize_embeddings=bool(
                embedding_config.get("normalize_embeddings", True)
            ),
        )

    if provider in {"openai_compatible", "openai"}:
        api_key_env = embedding_config.get("api_key_env", "EMBEDDING_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing embedding API key. Set environment variable {api_key_env}."
            )

        return OpenAICompatibleEmbeddings(
            model=embedding_config["model"],
            api_key=api_key,
            base_url=embedding_config.get("base_url"),
            batch_size=int(embedding_config.get("batch_size", 64)),
        )

    if provider == "bedrock":
        from langchain_community.embeddings.bedrock import BedrockEmbeddings

        return BedrockEmbeddings(
            credentials_profile_name=embedding_config.get("bedrock_profile", "default"),
            region_name=embedding_config.get("bedrock_region", "us-east-1"),
        )

    raise ValueError(
        f"Unsupported embedding provider '{provider}'. "
        "Use 'hashing', 'sentence_transformers', 'openai_compatible', 'openai', or 'bedrock'."
    )
