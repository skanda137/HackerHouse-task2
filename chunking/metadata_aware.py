"""
Strategy 3: Metadata-aware (language-routed) chunking.

MSMARCO-XI isn't one blob of text -- it's separate per-language configs
(hi, ta, te, bn, ...). That's a real, dataset-native axis to chunk on, not
a contrived one:

  - Indic scripts vary a lot in "information density per character".
    Tamil and Telugu are agglutinative and tend to run longer per sentence
    than Hindi/Bengali for the same amount of meaning, so a single fixed
    chunk_size across all languages either over-splits some languages or
    under-splits others.
  - Retrieval quality benefits from being able to filter/boost by language
    and by source/split at query time -- so every chunk this strategy
    produces carries rich metadata (language, source, dataset split,
    doc position, script) that the retriever and guardrail layer can use
    for filtering, not just for display.

This wrapper doesn't reinvent chunking math -- it *routes* each document to
a per-language-tuned underlying chunker (fixed-size or sentence-boundary)
and stamps the result with metadata. That routing + metadata tagging is the
actual "metadata-aware" contribution.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .base import BaseChunker, Chunk, Document
from .fixed_size import FixedSizeChunker
from .semantic import SentenceBoundaryChunker

# Rough per-language tuning. Indic languages with denser agglutination
# (Tamil, Telugu) get a larger target size so we don't fragment mid-thought;
# languages closer to Hindi/Bengali keep the default budget.
LANGUAGE_PROFILES: Dict[str, Dict[str, int]] = {
    "hi": {"target_size": 220, "max_size": 320},
    "bn": {"target_size": 220, "max_size": 320},
    "ta": {"target_size": 260, "max_size": 380},
    "te": {"target_size": 260, "max_size": 380},
    "en": {"target_size": 200, "max_size": 300},
}
DEFAULT_PROFILE = {"target_size": 220, "max_size": 320}

SCRIPT_BY_LANGUAGE = {
    "hi": "Devanagari", "bn": "Bengali", "ta": "Tamil",
    "te": "Telugu", "en": "Latin",
}


class MetadataAwareChunker(BaseChunker):
    name = "metadata_aware"

    def __init__(self, base_strategy: str = "semantic", extra_metadata: Optional[dict] = None):
        """
        base_strategy: "semantic" (sentence-boundary, language-tuned) or
                        "fixed" (fixed-size, language-tuned).
        """
        if base_strategy not in ("semantic", "fixed"):
            raise ValueError("base_strategy must be 'semantic' or 'fixed'")
        self.base_strategy = base_strategy
        self.extra_metadata = extra_metadata or {}
        self._chunker_cache: Dict[str, BaseChunker] = {}

    def _chunker_for_language(self, language: str) -> BaseChunker:
        if language in self._chunker_cache:
            return self._chunker_cache[language]
        profile = LANGUAGE_PROFILES.get(language, DEFAULT_PROFILE)
        if self.base_strategy == "semantic":
            c = SentenceBoundaryChunker(target_size=profile["target_size"], max_size=profile["max_size"])
        else:
            c = FixedSizeChunker(chunk_size=profile["target_size"], overlap=int(profile["target_size"] * 0.18))
        self._chunker_cache[language] = c
        return c

    def chunk(self, document: Document) -> List[Chunk]:
        underlying = self._chunker_for_language(document.language)
        chunks = underlying.chunk(document)
        for c in chunks:
            c.strategy = self.name
            c.metadata.update({
                "routed_via": underlying.name,
                "script": SCRIPT_BY_LANGUAGE.get(document.language, "unknown"),
                "language": document.language,
                "source": document.source,
                "doc_id": document.doc_id,
                **self.extra_metadata,
            })
        return chunks
