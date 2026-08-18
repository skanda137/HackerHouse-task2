"""
Retrieval layer -- this is the piece Person 2 owns the latency budget for.

`Retriever.retrieve()` times itself in two stages (query embedding, FAISS
search) and returns both the results and a `LatencyBreakdown` so the
benchmark script and the rest of the pipeline (guardrails, generation) can
report exactly where time went, not just a single end-to-end number.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import List, Optional, Tuple

from chunking.base import Chunk
from embeddings.embedder import BaseEmbedder
from index.faiss_index import FaissChunkIndex


@dataclass
class LatencyBreakdown:
    embed_ms: float
    search_ms: float

    @property
    def total_ms(self) -> float:
        return self.embed_ms + self.search_ms


@dataclass
class RetrievalResult:
    query: str
    hits: List[Tuple[Chunk, float]]
    latency: LatencyBreakdown


class Retriever:
    def __init__(self, embedder: BaseEmbedder, index: FaissChunkIndex, default_k: int = 5):
        self.embedder = embedder
        self.index = index
        self.default_k = default_k

    def retrieve(self, query: str, k: Optional[int] = None,
                 language: Optional[str] = None, strategy: Optional[str] = None) -> RetrievalResult:
        k = k or self.default_k

        t0 = perf_counter()
        qvec = self.embedder.encode_query(query)
        t1 = perf_counter()

        hits = self.index.search(qvec, k=k, language=language, strategy=strategy)
        t2 = perf_counter()

        latency = LatencyBreakdown(
            embed_ms=(t1 - t0) * 1000.0,
            search_ms=(t2 - t1) * 1000.0,
        )
        return RetrievalResult(query=query, hits=hits, latency=latency)
