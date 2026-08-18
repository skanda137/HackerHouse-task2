from .base import BaseChunker, Chunk, Document
from .fixed_size import FixedSizeChunker
from .semantic import SentenceBoundaryChunker, EmbeddingSemanticChunker, split_sentences
from .metadata_aware import MetadataAwareChunker

__all__ = [
    "BaseChunker", "Chunk", "Document",
    "FixedSizeChunker",
    "SentenceBoundaryChunker", "EmbeddingSemanticChunker", "split_sentences",
    "MetadataAwareChunker",
]

STRATEGY_REGISTRY = {
    "fixed_size": FixedSizeChunker,
    "semantic_sentence_boundary": SentenceBoundaryChunker,
    "metadata_aware": MetadataAwareChunker,
    # embedding-drift chunker takes an embedder at construction time,
    # so it's wired up explicitly in pipeline.py rather than via this registry.
}
