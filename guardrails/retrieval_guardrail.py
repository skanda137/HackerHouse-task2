"""
Retrieval-side guardrails.

The task requires the *system* to know when not to answer. Person 2's slice
of that is upstream of generation: if retrieval itself can't find anything
grounded, the generator should never even be invoked with false context.
This module turns raw FAISS scores into an explicit, structured verdict
that the generation/guardrail layer downstream consumes -- it should never
have to re-derive "was this actually relevant?" from a bare score.

Three checks, in order:

1. Empty result guard      -- no hits at all (empty/misconfigured index).
2. Top-score confidence     -- best cosine similarity below `min_top_score`
                                means nothing in the corpus is actually
                                relevant to the query (classic off-topic /
                                out-of-domain signal for a RAG system).
3. Score-gap sanity check   -- if the top hit isn't meaningfully better than
                                the rest, that's a sign of a diffuse, low-
                                confidence match rather than a real answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from chunking.base import Chunk


@dataclass
class RetrievalVerdict:
    should_answer: bool
    reason: str
    top_score: Optional[float]
    grounded_chunks: List[Chunk]


class RetrievalGuardrail:
    def __init__(self, min_top_score: float = 0.28, min_score_gap: float = 0.0):
        """
        min_top_score: cosine similarity floor. Below this, we treat the
            query as off-topic / unanswerable from the corpus.
        min_score_gap: optional minimum gap between top score and the
            (k)th score to flag "diffuse match, low confidence" -- defaults
            to 0 (disabled) since it's corpus-size dependent; set explicitly
            per deployment after looking at real score distributions.
        """
        self.min_top_score = min_top_score
        self.min_score_gap = min_score_gap

    def evaluate(self, hits: List[Tuple[Chunk, float]]) -> RetrievalVerdict:
        if not hits:
            return RetrievalVerdict(
                should_answer=False,
                reason="no_hits_returned",
                top_score=None,
                grounded_chunks=[],
            )

        top_chunk, top_score = hits[0]

        if top_score < self.min_top_score:
            return RetrievalVerdict(
                should_answer=False,
                reason=f"top_score_below_threshold ({top_score:.3f} < {self.min_top_score})",
                top_score=top_score,
                grounded_chunks=[],
            )

        if self.min_score_gap > 0 and len(hits) > 1:
            gap = top_score - hits[-1][1]
            if gap < self.min_score_gap:
                return RetrievalVerdict(
                    should_answer=False,
                    reason=f"diffuse_match_low_confidence (gap {gap:.3f})",
                    top_score=top_score,
                    grounded_chunks=[],
                )

        # Answerable: hand downstream only the chunks that clear the bar,
        # not the full over-fetched candidate list.
        grounded = [c for c, s in hits if s >= self.min_top_score]
        return RetrievalVerdict(
            should_answer=True,
            reason="ok",
            top_score=top_score,
            grounded_chunks=grounded,
        )
