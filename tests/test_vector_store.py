import tempfile
from pathlib import Path

from triage_agent.embeddings import Embedder
from triage_agent.vector_store import VectorStore


def _tmp_store() -> VectorStore:
    return VectorStore(path=str(Path(tempfile.mkdtemp()) / "chroma"))


def test_upsert_and_query_round_trip():
    store = _tmp_store()
    e = Embedder()
    store.upsert(
        "owner/repo", 1, e.embed("memory leak in middleware"),
        document="memory leak in middleware", metadata={"title": "Leak", "priority": "P1", "issue_type": "bug", "reasoning": "r"},
    )
    store.upsert(
        "owner/repo", 2, e.embed("typo in readme"),
        document="typo in readme", metadata={"title": "Typo", "priority": "P3", "issue_type": "documentation", "reasoning": "r2"},
    )

    hits = store.query("owner/repo", e.embed("middleware leaking memory over time"), top_k=1)
    assert hits[0]["issue_number"] == 1


def test_query_empty_collection_returns_empty():
    store = _tmp_store()
    e = Embedder()
    assert store.query("owner/nothing-indexed", e.embed("anything"), top_k=3) == []


def test_repos_are_isolated():
    store = _tmp_store()
    e = Embedder()
    store.upsert("repo-a", 1, e.embed("issue in repo a"), document="a", metadata={"title": "A", "priority": "P1", "issue_type": "bug", "reasoning": "r"})
    hits = store.query("repo-b", e.embed("issue in repo a"), top_k=3)
    assert hits == []
