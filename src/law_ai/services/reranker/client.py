"""In-process cross-encoder reranker (sentence-transformers)."""

import asyncio
from contextlib import nullcontext
from functools import partial
from typing import Any

from law_ai.config import RerankerSettings
from law_ai.schemas.chunk import RetrievedChunk
from law_ai.services.reranker.base import BaseReranker


class CrossEncoderReranker(BaseReranker):
    def __init__(self, settings: RerankerSettings, tracer: Any = None) -> None:
        self._settings = settings
        self._tracer = tracer
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        if not self._settings.model or not candidates:
            return candidates[:top_k] if top_k is not None else candidates

        span_cm = (
            self._tracer.span(
                "rerank",
                input={"query": query, "candidates": len(candidates), "top_k": top_k},
            )
            if self._tracer is not None
            else nullcontext()
        )
        with span_cm as span:
            model = await self._ensure_model()
            pairs = [(query, c.chunk.text) for c in candidates]
            rrf_rank = {c.chunk.chunk_id: i for i, c in enumerate(candidates, start=1)}
            rrf_score = {c.chunk.chunk_id: c.score for c in candidates}

            scores = await asyncio.to_thread(
                partial(model.predict, pairs, batch_size=self._settings.batch_size)
            )
            reranked = sorted(
                zip(candidates, scores, strict=True), key=lambda item: item[1], reverse=True
            )
            scored = [RetrievedChunk(chunk=c.chunk, score=float(s)) for c, s in reranked]
            results = scored[:top_k] if top_k is not None else scored

            if span is not None:
                span.update(
                    output={
                        "model": self._settings.model,
                        "scored_pairs": len(pairs),
                        "batch_size": self._settings.batch_size,
                        "results": [
                            {
                                "chunk_id": r.chunk.chunk_id,
                                "article": r.chunk.metadata.article,
                                "act": r.chunk.metadata.act,
                                "rerank_score": round(r.score, 5),
                                "rrf_rank": rrf_rank.get(r.chunk.chunk_id),
                                "rrf_score": round(rrf_score.get(r.chunk.chunk_id, 0.0), 5),
                                "text_preview": r.chunk.text[:200],
                            }
                            for r in results
                        ],
                    }
                )
            return results

    async def _ensure_model(self) -> Any:
        if self._model is None:
            async with self._lock:
                if self._model is None:

                    def _load() -> Any:
                        from sentence_transformers import CrossEncoder

                        kwargs: dict[str, Any] = {}
                        if self._settings.device:
                            kwargs["device"] = self._settings.device
                        if self._settings.fp16:
                            import torch

                            kwargs["model_kwargs"] = {"dtype": torch.float16}
                        return CrossEncoder(self._settings.model, **kwargs)

                    self._model = await asyncio.to_thread(_load)
        return self._model

    async def warmup(self) -> None:
        """Pre-load so the first query does not pay the model load.

        Non-fatal: missing weights should not take the stack down — the lazy
        path retries on first use.
        """
        if self._settings.model:
            model = await self._ensure_model()
            await asyncio.to_thread(model.predict, [("warmup", "warmup")])
