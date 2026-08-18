"""
Strategy 1: Fixed-size chunking with overlap.

The simplest strategy, and the one the task brief explicitly warns against
using *alone*. We keep it because it's a fast, predictable baseline that's
useful for very short or already-clean passages (MSMARCO passages are often
just 1-3 sentences) -- but we never ship it as the only strategy.

Overlap matters here: without it, a fact that straddles a chunk boundary
becomes unretrievable no matter how good the embedding model is. We default
to ~15% overlap, tunable per call.
"""
from __future__ import annotations

from typing import List

from .base import BaseChunker, Chunk, Document


class FixedSizeChunker(BaseChunker):
    name = "fixed_size"

    def __init__(self, chunk_size: int = 220, overlap: int = 40):
        """
        chunk_size: max characters per chunk.
        overlap: characters repeated between consecutive chunks.
        """
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> List[Chunk]:
        text = document.text.strip()
        if not text:
            return []

        step = self.chunk_size - self.overlap
        chunks: List[Chunk] = []
        position = 0
        start = 0
        n = len(text)

        # Short passages (very common in MSMARCO) don't need splitting at all --
        # emit a single chunk rather than padding logic for no reason.
        if n <= self.chunk_size:
            return [Chunk(
                chunk_id=Chunk.make_id(document.doc_id, self.name, 0),
                text=text,
                doc_id=document.doc_id,
                language=document.language,
                strategy=self.name,
                position=0,
                char_start=0,
                char_end=n,
                metadata={"source": document.source, "chunk_size": self.chunk_size, "overlap": self.overlap},
            )]

        while start < n:
            end = min(start + self.chunk_size, n)
            piece = text[start:end]
            chunks.append(Chunk(
                chunk_id=Chunk.make_id(document.doc_id, self.name, position),
                text=piece,
                doc_id=document.doc_id,
                language=document.language,
                strategy=self.name,
                position=position,
                char_start=start,
                char_end=end,
                metadata={"source": document.source, "chunk_size": self.chunk_size, "overlap": self.overlap},
            ))
            position += 1
            if end == n:
                break
            start += step

        return chunks
