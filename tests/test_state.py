import tempfile
from pathlib import Path

from triage_agent.embeddings import Embedder
from triage_agent.state import TriageRecord, TriageState
from triage_agent.vector_store import VectorStore


def _tmp_state() -> TriageState:
    tmp = Path(tempfile.mkdtemp()) / "state.json"
    return TriageState(tmp)


def _tmp_state_with_vectors() -> TriageState:
    tmp_dir = Path(tempfile.mkdtemp())
    return TriageState(
        tmp_dir / "state.json",
        vector_store=VectorStore(str(tmp_dir / "chroma")),
        embedder=Embedder(),
    )


def test_record_and_already_triaged():
    state = _tmp_state()
    repo = "owner/repo"
    assert not state.already_triaged(repo, 1)

    state.record(repo, TriageRecord(
        issue_number=1, title="Login button broken on Safari",
        priority="P1", issue_type="bug", confidence=0.9,
        labels_suggested=["bug", "ui"], draft_response="Thanks for reporting...",
        reasoning="Reproducible browser-specific bug.",
    ))
    assert state.already_triaged(repo, 1)
    assert not state.already_triaged(repo, 2)


def test_context_summary_reflects_history():
    state = _tmp_state()
    repo = "owner/repo"
    state.record(repo, TriageRecord(
        issue_number=1, title="Crash on startup", priority="P0", issue_type="bug",
        confidence=0.95, labels_suggested=["bug", "crash"], draft_response="...", reasoning="...",
    ))
    summary = state.context_summary(repo)
    assert "Triaged so far: 1" in summary
    assert "P0" in summary


def test_find_similar_uses_keyword_overlap():
    state = _tmp_state()
    repo = "owner/repo"
    state.record(repo, TriageRecord(
        issue_number=1, title="Memory leak in middleware chain", priority="P1",
        issue_type="bug", confidence=0.8, labels_suggested=["bug", "performance"],
        draft_response="...", reasoning="Similar to known middleware GC issue.",
    ))
    state.record(repo, TriageRecord(
        issue_number=2, title="Typo in README installation section", priority="P3",
        issue_type="documentation", confidence=0.99, labels_suggested=["docs"],
        draft_response="...", reasoning="Trivial docs fix.",
    ))

    similar = state.find_similar(repo, "Memory leak when using nested middleware")
    assert similar
    assert similar[0]["issue_number"] == 1  # the middleware/memory issue, not the docs typo


def test_context_summary_empty_state():
    state = _tmp_state()
    summary = state.context_summary("owner/repo")
    assert "No prior triage history" in summary


def test_find_similar_semantic_falls_back_without_vector_store():
    state = _tmp_state()  # no vector_store/embedder configured
    repo = "owner/repo"
    state.record(repo, TriageRecord(
        issue_number=1, title="Memory leak in middleware chain", priority="P1",
        issue_type="bug", confidence=0.8, labels_suggested=["bug"],
        draft_response="...", reasoning="...",
    ))
    similar = state.find_similar_semantic(repo, "Memory leak when using nested middleware")
    assert similar and similar[0]["issue_number"] == 1


def test_find_similar_semantic_uses_vector_store_when_configured():
    state = _tmp_state_with_vectors()
    repo = "owner/repo"
    state.record(
        repo,
        TriageRecord(
            issue_number=1, title="Memory leak", priority="P1", issue_type="bug",
            confidence=0.8, labels_suggested=["bug"], draft_response="...",
            reasoning="Confirmed leak in middleware chain.",
        ),
        index_text="Memory leak in middleware chain when using async handlers repeatedly",
    )
    state.record(
        repo,
        TriageRecord(
            issue_number=2, title="Docs typo", priority="P3", issue_type="documentation",
            confidence=0.9, labels_suggested=["docs"], draft_response="...", reasoning="Trivial.",
        ),
        index_text="Typo in the README installation section",
    )

    similar = state.find_similar_semantic(repo, "middleware causes memory to grow unbounded")
    assert similar[0]["issue_number"] == 1
