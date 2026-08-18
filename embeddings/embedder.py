"""
Pluggable embedding layer.

Two implementations, same interface (`.encode(texts) -> np.ndarray`):

1. `SentenceTransformerEmbedder` -- the real, production embedder. Uses a
   multilingual model (default: intfloat/multilingual-e5-small) so hi/ta/te/bn
   all land in the same vector space. This is what the submitted system
   should run with -- it requires downloading model weights from the
   HuggingFace Hub, so it needs outbound network access once, on first run
   (weights are then cached locally and every later run is fully offline).

2. `TfidfEmbedder` -- a pure scikit-learn, zero-download fallback. It exists
   for two reasons: (a) CI / offline dev boxes without HF Hub access can
   still run the full pipeline and tests, and (b) it's a useful latency
   floor to compare against -- if TF-IDF retrieval is already borderline on
   the 200ms budget, that tells us where the budget is actually going.

Both are normalized to unit length so cosine similarity == dot product,
which lets the FAISS index always use IndexFlatIP regardless of which
embedder produced the vectors.
"""
from __future__ import annotations

from typing import List, Optional
import numpy as np


class BaseEmbedder:
    dim: int
    name: str = "base"

    def encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_query(self, text: str) -> np.ndarray:
        """Some models (e.g. e5) want a different prefix for queries vs.
        passages. Default: same as encode()."""
        return self.encode([text])[0]


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Production embedder. Requires `pip install sentence-transformers` and
    (on first run only) network access to the HuggingFace Hub to download
    weights -- after that, everything is local and offline.
    """
    name = "sentence_transformer"

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        from sentence_transformers import SentenceTransformer  # lazy import
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()
        # e5 models are trained with "query: " / "passage: " instruction
        # prefixes -- using them measurably improves retrieval quality.
        self._uses_e5_prefix = "e5" in model_name.lower()

    def encode(self, texts: List[str]) -> np.ndarray:
        prefixed = [f"passage: {t}" if self._uses_e5_prefix else t for t in texts]
        vecs = self._model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")

    def encode_query(self, text: str) -> np.ndarray:
        prefixed = f"query: {text}" if self._uses_e5_prefix else text
        vec = self._model.encode([prefixed], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.asarray(vec, dtype="float32")


class TfidfEmbedder(BaseEmbedder):
    """
    Offline fallback embedder -- no network, no model download. Fits a
    character n-gram TF-IDF vectorizer over the corpus (char n-grams, not
    word n-grams, because word tokenization differs a lot across
    Devanagari/Tamil/Telugu/Bengali scripts and char n-grams are far more
    script-agnostic without language-specific tokenizers).

    Must be `.fit(corpus_texts)` once before `.encode()` is used.
    """
    name = "tfidf_char_ngram"

    def __init__(self, max_features: int = 20000, ngram_range=(2, 4)):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=ngram_range, max_features=max_features,
        )
        self._fitted = False
        self.dim: Optional[int] = None

    def fit(self, corpus_texts: List[str]):
        self.vectorizer.fit(corpus_texts)
        self._fitted = True
        self.dim = len(self.vectorizer.vocabulary_)
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder must be .fit(corpus) before encode()")
        mat = self.vectorizer.transform(texts).toarray().astype("float32")
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms
