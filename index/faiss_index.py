"""
FAISS was picked over a hosted vector DB (Pinecone/Weaviate Cloud/etc.)
specifically because the task caps end-to-end latency at 200ms. A hosted
DB adds a network round-trip that's entirely outside our control and
frequently *alone* exceeds 200ms from a laptop/CI runner. FAISS runs
in-process, in-memory -- the only latency is CPU math, which is exactly
what we can profile and optimize.

Index choice: `IndexFlatIP` (exact, brute-force inner product) for corpora
in the tens of thousands of chunks -- which is the realistic scale here.
Exact search avoids recall loss from approximate indexes (HNSW/IVF) and is
still comfortably sub-200ms at this scale. The wrapper exposes a one-line
swap to `IndexIVFFlat` for when the corpus grows large enough that exact
search stops being the right tradeoff -- see `build_ivf`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import faiss

from chunking.base import Chunk


class FaissChunkIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self._chunks: List[Chunk] = []  # position i in this list <-> row i in the index

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add(self, chunks: List[Chunk], vectors: np.ndarray):
        assert vectors.shape[0] == len(chunks)
        assert vectors.shape[1] == self.dim
        self.index.add(vectors.astype("float32"))
        self._chunks.extend(chunks)

    def build_ivf(self, nlist: int = 100):
        """
        Swap the flat index for an IVF index once the corpus is large enough
        that exact search is no longer the right tradeoff (rule of thumb:
        low hundreds of thousands of vectors and up, well beyond MSMARCO-XI's
        realistic per-language chunk counts for this task). Rebuilds the
        index from currently-stored vectors.
        """
        if self.size == 0:
            raise RuntimeError("nothing indexed yet")
        vecs = self.index.reconstruct_n(0, self.size)
        quantizer = faiss.IndexFlatIP(self.dim)
        ivf = faiss.IndexIVFFlat(quantizer, self.dim, min(nlist, max(1, self.size // 4)))
        ivf.train(vecs)
        ivf.add(vecs)
        ivf.nprobe = 8
        self.index = ivf

    def search(self, query_vec: np.ndarray, k: int = 5,
               language: Optional[str] = None,
               strategy: Optional[str] = None) -> List[Tuple[Chunk, float]]:
        """
        Metadata filtering (language / strategy) is applied by over-fetching
        from FAISS and filtering in Python -- simple, and fast enough at this
        corpus scale. At much larger scale this would move to FAISS's native
        metadata/ID-selector filtering or a filtered-HNSW index instead.
        """
        query_vec = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        needs_filter = language is not None or strategy is not None
        fetch_k = k * 8 if needs_filter else k
        fetch_k = min(fetch_k, self.size) if self.size else 0
        if fetch_k == 0:
            return []

        scores, idxs = self.index.search(query_vec, fetch_k)
        results: List[Tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = self._chunks[idx]
            if language is not None and chunk.language != language:
                continue
            if strategy is not None and chunk.strategy != strategy:
                continue
            results.append((chunk, float(score)))
            if len(results) >= k:
                break
        return results
