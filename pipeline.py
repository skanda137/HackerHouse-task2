"""
Person 2's harnessed pipeline: dataset -> chunking (3 strategies) ->
embedding -> FAISS index -> retrieval -> retrieval guardrail.

"Harnessed" here means: every external/fallible step (embedding a batch,
building the index, running a search) goes through explicit retry +
structured error handling rather than being a bare call that can crash the
whole run. Every stage returns a structured result (dataclass), never a
raw string, so the generation component (owned by another teammate) has a
typed contract to build against.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from chunking.base import Chunk, Document
from chunking.fixed_size import FixedSizeChunker
from chunking.semantic import SentenceBoundaryChunker, EmbeddingSemanticChunker
from chunking.metadata_aware import MetadataAwareChunker
from embeddings.embedder import BaseEmbedder, TfidfEmbedder
from index.faiss_index import FaissChunkIndex
from retrieval.retriever import Retriever, RetrievalResult
from guardrails.retrieval_guardrail import RetrievalGuardrail, RetrievalVerdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pipeline")


def retry(fn, *args, attempts: int = 3, backoff_s: float = 0.2, **kwargs):
    """Small generic retry wrapper -- used around every fallible external
    call (embedding batches, index ops) so a single transient failure
    doesn't take down the whole indexing/retrieval run."""
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- intentionally broad: this is a generic harness
            last_exc = exc
            logger.warning("attempt %d/%d failed for %s: %s", attempt, attempts, getattr(fn, "__name__", fn), exc)
            if attempt < attempts:
                time.sleep(backoff_s * attempt)
    raise RuntimeError(f"all {attempts} attempts failed") from last_exc


@dataclass
class IndexBuildReport:
    strategy: str
    n_documents: int
    n_chunks: int
    build_ms: float
    languages: List[str] = field(default_factory=list)


@dataclass
class QueryOutcome:
    """Structured, typed output of a single end-to-end retrieval call --
    this is the contract handed to the generation component."""
    query: str
    verdict: RetrievalVerdict
    embed_ms: float
    search_ms: float
    total_ms: float
    strategy: str


class RetrievalPipeline:
    def __init__(self, embedder: Optional[BaseEmbedder] = None,
                 guardrail: Optional[RetrievalGuardrail] = None):
        self.embedder = embedder or TfidfEmbedder()
        self.guardrail = guardrail or RetrievalGuardrail()
        self.indexes: dict[str, FaissChunkIndex] = {}       # strategy_name -> index
        self.chunk_counts: dict[str, int] = {}
        self.build_reports: List[IndexBuildReport] = []

    # -- Stage 1: chunking -------------------------------------------------
    def build_chunks(self, documents: List[Document]) -> dict[str, List[Chunk]]:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=220, overlap=40),
            "semantic_sentence_boundary": SentenceBoundaryChunker(target_size=220, max_size=320),
            "metadata_aware": MetadataAwareChunker(base_strategy="semantic"),
        }
        result: dict[str, List[Chunk]] = {}
        for name, chunker in strategies.items():
            t0 = time.perf_counter()
            chunks = chunker.chunk_many(documents)
            dt = (time.perf_counter() - t0) * 1000.0
            result[name] = chunks
            logger.info("[chunking] %-28s -> %5d chunks from %4d docs (%.1fms)",
                        name, len(chunks), len(documents), dt)
        return result

    # -- Stage 2+3: embed + index (with retry) ------------------------------
    def build_index(self, strategy: str, chunks: List[Chunk]) -> IndexBuildReport:
        if not chunks:
            raise ValueError(f"no chunks to index for strategy={strategy}")

        texts = [c.text for c in chunks]

        if isinstance(self.embedder, TfidfEmbedder) and not self.embedder._fitted:
            retry(self.embedder.fit, texts)

        t0 = time.perf_counter()
        vectors = retry(self.embedder.encode, texts)
        vectors = np.asarray(vectors, dtype="float32")

        index = FaissChunkIndex(dim=vectors.shape[1])
        retry(index.add, chunks, vectors)
        build_ms = (time.perf_counter() - t0) * 1000.0

        self.indexes[strategy] = index
        self.chunk_counts[strategy] = len(chunks)
        report = IndexBuildReport(
            strategy=strategy, n_documents=len({c.doc_id for c in chunks}),
            n_chunks=len(chunks), build_ms=build_ms,
            languages=sorted({c.language for c in chunks}),
        )
        self.build_reports.append(report)
        logger.info("[index] %-28s -> %5d vectors indexed (%.1fms)", strategy, len(chunks), build_ms)
        return report

    def build_all(self, documents: List[Document]):
        chunk_sets = self.build_chunks(documents)
        # TF-IDF must be fit on the union of all chunk texts once, so every
        # strategy's index lives in the same vector space and scores are
        # comparable across strategies.
        if isinstance(self.embedder, TfidfEmbedder) and not self.embedder._fitted:
            all_texts = [c.text for chunks in chunk_sets.values() for c in chunks]
            retry(self.embedder.fit, all_texts)
        for strategy, chunks in chunk_sets.items():
            self.build_index(strategy, chunks)
        return self.build_reports

    # -- Stage 4: retrieve + guardrail (harnessed) --------------------------
    def query(self, text: str, strategy: str = "metadata_aware", k: int = 5,
              language: Optional[str] = None) -> QueryOutcome:
        if strategy not in self.indexes:
            raise KeyError(f"strategy '{strategy}' has not been indexed yet. "
                            f"Available: {list(self.indexes.keys())}")

        retriever = Retriever(self.embedder, self.indexes[strategy], default_k=k)
        try:
            result: RetrievalResult = retry(retriever.retrieve, text, k=k, language=language, attempts=2)
        except Exception as exc:
            logger.error("retrieval failed for query=%r: %s", text, exc)
            verdict = RetrievalVerdict(should_answer=False, reason=f"retrieval_error: {exc}",
                                        top_score=None, grounded_chunks=[])
            return QueryOutcome(query=text, verdict=verdict, embed_ms=0.0, search_ms=0.0,
                                 total_ms=0.0, strategy=strategy)

        verdict = self.guardrail.evaluate(result.hits)
        return QueryOutcome(
            query=text, verdict=verdict,
            embed_ms=result.latency.embed_ms, search_ms=result.latency.search_ms,
            total_ms=result.latency.total_ms, strategy=strategy,
        )
