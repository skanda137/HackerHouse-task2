"""
Strategy 2: Semantic / sentence-boundary chunking.

Two variants live here:

1. `SentenceBoundaryChunker` -- splits on real sentence boundaries (handling
   both Latin punctuation `. ! ?` and the Indic danda `। ॥` used across
   Hindi/Bengali/etc.), then greedily packs sentences into a chunk until a
   target size budget is hit. This never cuts a sentence in half, which
   fixed-size chunking can do.

2. `EmbeddingSemanticChunker` -- a step further: sentences are embedded and
   merged based on cosine-similarity drift. When the next sentence's meaning
   diverges from the running chunk's centroid past a threshold, we close the
   chunk and start a new one. This is closer to genuine "semantic" chunking
   (topic-shift aware) rather than just size-budgeted sentence packing. It's
   pluggable with any embedder that implements `.encode(texts) -> np.ndarray`.
"""
from __future__ import annotations

import re
from typing import List, Optional

import numpy as np

from .base import BaseChunker, Chunk, Document

# Sentence terminators: Latin ., !, ? and the Devanagari/Indic danda ।, ॥
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।॥])\s+")


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


class SentenceBoundaryChunker(BaseChunker):
    name = "semantic_sentence_boundary"

    def __init__(self, target_size: int = 220, max_size: int = 320):
        self.target_size = target_size
        self.max_size = max_size

    def chunk(self, document: Document) -> List[Chunk]:
        sentences = split_sentences(document.text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        buf: List[str] = []
        buf_len = 0
        position = 0
        cursor = 0  # tracks char offset into the original text for provenance

        def flush(end_cursor: int):
            nonlocal position
            if not buf:
                return
            piece = " ".join(buf)
            chunks.append(Chunk(
                chunk_id=Chunk.make_id(document.doc_id, self.name, position),
                text=piece,
                doc_id=document.doc_id,
                language=document.language,
                strategy=self.name,
                position=position,
                char_start=max(0, end_cursor - len(piece)),
                char_end=end_cursor,
                metadata={"source": document.source, "sentence_count": len(buf)},
            ))
            position += 1

        for sent in sentences:
            cursor += len(sent) + 1
            # A single sentence longer than max_size still gets its own chunk
            # (better an oversized-but-intact chunk than a broken one).
            if buf_len + len(sent) > self.max_size and buf:
                flush(cursor - len(sent) - 1)
                buf, buf_len = [], 0
            buf.append(sent)
            buf_len += len(sent) + 1
            if buf_len >= self.target_size:
                flush(cursor)
                buf, buf_len = [], 0

        flush(cursor)
        return chunks


class EmbeddingSemanticChunker(BaseChunker):
    """
    Topic-shift-aware chunking: merges consecutive sentences while they stay
    semantically close to the chunk's running centroid, and cuts a new chunk
    when similarity drops below `similarity_threshold`.

    Requires an embedder exposing `.encode(List[str]) -> np.ndarray` (see
    embeddings/embedder.py). Falls back gracefully to sentence packing if a
    document has too few sentences to make semantic grouping meaningful.
    """
    name = "semantic_embedding_drift"

    def __init__(self, embedder, similarity_threshold: float = 0.55,
                 min_sentences: int = 2, max_size: int = 400):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.min_sentences = min_sentences
        self.max_size = max_size

    @staticmethod
    def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def chunk(self, document: Document) -> List[Chunk]:
        sentences = split_sentences(document.text)
        if len(sentences) < self.min_sentences:
            # Too short to benefit from drift detection -- one chunk, whole doc.
            if not sentences:
                return []
            text = " ".join(sentences)
            return [Chunk(
                chunk_id=Chunk.make_id(document.doc_id, self.name, 0),
                text=text, doc_id=document.doc_id, language=document.language,
                strategy=self.name, position=0, char_start=0, char_end=len(text),
                metadata={"source": document.source, "sentence_count": len(sentences), "note": "too_short_for_drift"},
            )]

        vecs = self.embedder.encode(sentences)
        chunks: List[Chunk] = []
        buf_sents = [sentences[0]]
        buf_vecs = [vecs[0]]
        position = 0
        char_cursor = 0

        def centroid(vs):
            return np.mean(np.stack(vs), axis=0)

        def flush():
            nonlocal position
            piece = " ".join(buf_sents)
            chunks.append(Chunk(
                chunk_id=Chunk.make_id(document.doc_id, self.name, position),
                text=piece, doc_id=document.doc_id, language=document.language,
                strategy=self.name, position=position,
                char_start=char_cursor, char_end=char_cursor + len(piece),
                metadata={"source": document.source, "sentence_count": len(buf_sents)},
            ))
            position += 1

        for sent, vec in zip(sentences[1:], vecs[1:]):
            sim = self._cos_sim(centroid(buf_vecs), vec)
            proposed_len = sum(len(s) for s in buf_sents) + len(sent)
            if sim < self.similarity_threshold or proposed_len > self.max_size:
                flush()
                char_cursor += sum(len(s) + 1 for s in buf_sents)
                buf_sents, buf_vecs = [sent], [vec]
            else:
                buf_sents.append(sent)
                buf_vecs.append(vec)

        flush()
        return chunks
