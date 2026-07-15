"""Embedder implementations.

- LocalEmbedder: sentence-transformers in-process (needs the `ml` dep group).
  Model loading and encode() are offloaded to a thread to keep the loop free.
- APIEmbedder: HTTP endpoint speaking the TEI (text-embeddings-inference)
  protocol — use when embeddings are served remotely.
"""

import asyncio
from functools import partial
from typing import Any

import httpx

from law_ai.config import EmbeddingSettings
from law_ai.services.embedding.base import BaseEmbedder


class LocalEmbedder(BaseEmbedder):
    def __init__(self, settings: EmbeddingSettings) -> None:
        self._settings = settings
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def _ensure_model(self) -> Any:
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    self._model = await asyncio.to_thread(self._load)
        return self._model

    def _load(self) -> Any:
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        return SentenceTransformer(self._settings.model)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model = await self._ensure_model()
        vectors = await asyncio.to_thread(partial(model.encode, texts, normalize_embeddings=True))
        return [v.tolist() for v in vectors]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_texts([text]))[0]

    @property
    def dimension(self) -> int:
        return self._settings.dimension


class APIEmbedder(BaseEmbedder):
    def __init__(self, settings: EmbeddingSettings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.api_url, timeout=settings.timeout_seconds
        )

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # whole acts are embedded at ingest — batch the requests
        batch_size = self._settings.batch_size
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), batch_size):
            batch = texts[offset : offset + batch_size]
            response = await self._client.post("/embed", json={"inputs": batch})
            response.raise_for_status()
            vectors.extend(response.json())
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_texts([text]))[0]

    @property
    def dimension(self) -> int:
        return self._settings.dimension

    async def aclose(self) -> None:
        await self._client.aclose()
