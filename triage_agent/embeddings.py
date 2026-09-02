"""Text embedder for the RAG layer.

Uses scikit-learn's HashingVectorizer: a deterministic, fixed-dimension
hash-trick embedding. No pretrained weights to download, no network
dependency, no vocabulary drift as the corpus grows over time. This is
a legitimate baseline retrieval strategy (BM25/TF-IDF-family), not a
placeholder -- swapping in a neural embedding model later means
replacing this one class, since VectorStore only ever consumes
List[float].
"""
from __future__ import annotations

from sklearn.feature_extraction.text import HashingVectorizer

DIMENSIONS = 256


class Embedder:
    def __init__(self, n_features: int = DIMENSIONS):
        self._vectorizer = HashingVectorizer(
            n_features=n_features, norm="l2", alternate_sign=False, stop_words="english"
        )

    def embed(self, text: str) -> list[float]:
        vec = self._vectorizer.transform([text])
        return vec.toarray()[0].tolist()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        vecs = self._vectorizer.transform(texts)
        return vecs.toarray().tolist()
