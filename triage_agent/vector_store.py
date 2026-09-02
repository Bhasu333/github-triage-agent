"""Persistent vector store for triaged-issue retrieval, one Chroma
collection per repo so different repos' histories never mix."""
from __future__ import annotations

import re

import chromadb


def _collection_name(repo_key: str) -> str:
    # Chroma collection names are restricted; sanitize owner/repo -> owner__repo
    return re.sub(r"[^a-zA-Z0-9_-]", "_", repo_key)


class VectorStore:
    def __init__(self, path: str = "chroma_store"):
        self._client = chromadb.PersistentClient(path=path)

    def _collection(self, repo_key: str):
        return self._client.get_or_create_collection(_collection_name(repo_key))

    def upsert(self, repo_key: str, issue_number: int, embedding: list[float], document: str, metadata: dict) -> None:
        col = self._collection(repo_key)
        col.upsert(ids=[str(issue_number)], embeddings=[embedding], documents=[document], metadatas=[metadata])

    def query(self, repo_key: str, embedding: list[float], top_k: int = 3) -> list[dict]:
        col = self._collection(repo_key)
        if col.count() == 0:
            return []
        top_k = min(top_k, col.count())
        result = col.query(query_embeddings=[embedding], n_results=top_k)
        hits = []
        for doc, meta, dist, doc_id in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0], result["ids"][0]
        ):
            hits.append({"issue_number": int(doc_id), "document": doc, "metadata": meta, "distance": dist})
        return hits
