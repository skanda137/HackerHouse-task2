import re
import time
from typing import List, Tuple
from app.schemas import RetrievedChunk


class LightweightGroundingEngine:
    def __init__(self, token_overlap_threshold: float = 0.60):
        self.threshold = token_overlap_threshold
        # Indic and alphanumeric tokenizer regex
        self.tokenizer = re.compile(r"[\w\u0900-\u0D7F]+", re.UNICODE)
        
        # Stopwords across English and major Indic languages (Hindi, Tamil, Telugu, Bengali)
        self.stopwords = {
            "is", "the", "are", "was", "were", "and", "in", "to", "of", "a", "an", "that", "it",
            "है", "हैं", "था", "थी", "थे", "का", "के", "की", "में", "पर", "और", "से", "को", "यह", "वह",
            "ஆகும்", "உள்ளது", "மற்றும்", "இல்", "ஒரு",
            "ఉంది", "మరియు", "లో", "ఒక",
            "হয়", "এবং", "একটি", "তে", "এ"
        }

    def _tokenize(self, text: str) -> set:
        tokens = self.tokenizer.findall(text.lower())
        return {t for t in tokens if t not in self.stopwords and len(t) > 1}

    def verify_grounding(self, answer: str, context_chunks: List[RetrievedChunk]) -> Tuple[bool, float, float]:
        """
        Returns:
            - is_grounded (bool)
            - overlap_ratio (float)
            - execution_latency_ms (float)
        """
        start_time = time.perf_counter()
        
        if not answer or not context_chunks:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return False, 0.0, elapsed

        # Extract answer content words
        answer_tokens = self._tokenize(answer)
        if not answer_tokens:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return True, 1.0, elapsed

        # Aggregate context tokens
        context_corpus = " ".join([c.text for c in context_chunks])
        context_tokens = self._tokenize(context_corpus)

        # Calculate directional overlap: % of claims in answer found in context
        supported_tokens = answer_tokens.intersection(context_tokens)
        overlap_ratio = len(supported_tokens) / len(answer_tokens)

        is_grounded = overlap_ratio >= self.threshold
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return is_grounded, round(overlap_ratio, 4), round(elapsed, 2)