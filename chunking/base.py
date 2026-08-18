"""
Base interfaces for the chunking layer.

Every chunking strategy in this package implements `BaseChunker.chunk(document)`
and returns a list of `Chunk` objects. Keeping a shared `Chunk` schema means the
embedding/index/retrieval layers downstream never need to know which strategy
produced a given chunk.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import hashlib


@dataclass
class Document:
    """A single raw record from the dataset before chunking."""
    doc_id: str
    text: str
    language: str                      # e.g. "hi", "ta", "te", "bn"
    source: str = "msmarco-xi"         # passage source / split name
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable unit produced by a chunker."""
    chunk_id: str
    text: str
    doc_id: str
    language: str
    strategy: str                      # which chunking strategy produced this
    position: int                      # order of this chunk within its document
    char_start: int
    char_end: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make_id(doc_id: str, strategy: str, position: int) -> str:
        raw = f"{doc_id}::{strategy}::{position}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class BaseChunker:
    """All chunking strategies subclass this."""
    name: str = "base"

    def chunk(self, document: Document) -> List[Chunk]:
        raise NotImplementedError

    def chunk_many(self, documents: List[Document]) -> List[Chunk]:
        out: List[Chunk] = []
        for doc in documents:
            out.extend(self.chunk(doc))
        return out
