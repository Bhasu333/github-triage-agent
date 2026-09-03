import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from triage_agent.github_client import Issue
from triage_agent.state import TriageRecord, TriageState
from triage_agent.tools import ToolDispatcher


def _tmp_state() -> TriageState:
    tmp = Path(tempfile.mkdtemp()) / "state.json"
    return TriageState(tmp)


def _fake_issue() -> Issue:
    return Issue(
        number=101, title="Memory leak in middleware", body="Leaks on v4.19.x",
        labels=["needs-triage"], author="demo-reporter", created_at="2026-08-20T10:00:00Z",
        comments_count=1, html_url="https://github.com/x/y/issues/101",
    )


def test_get_issue_details_dispatches_to_github_client():
    github = MagicMock()
    github.get_issue.return_value = _fake_issue()
    github.get_issue_comments.return_value = ["Can repro on my end too."]
    state = _tmp_state()

    dispatcher = ToolDispatcher(github, state, "owner", "repo", 101, "Memory leak in middleware")
    result = dispatcher.dispatch("get_issue_details", {})

    github.get_issue.assert_called_once_with("owner", "repo", 101)
    assert "Memory leak in middleware" in result
    assert "Can repro on my end too." in result


def test_get_similar_past_triages_includes_history():
    github = MagicMock()
    state = _tmp_state()
    state.record("owner/repo", TriageRecord(
        issue_number=1, title="Memory leak in async handlers", priority="P1",
        issue_type="bug", confidence=0.85, labels_suggested=["bug"],
        draft_response="...", reasoning="Confirmed leak, same root cause pattern.",
    ))

    dispatcher = ToolDispatcher(github, state, "owner", "repo", 101, "Memory leak in middleware")
    result = dispatcher.dispatch("get_similar_past_triages", {})

    assert "#1" in result
    assert "Confirmed leak" in result


def test_unknown_tool_raises():
    github = MagicMock()
    state = _tmp_state()
    dispatcher = ToolDispatcher(github, state, "owner", "repo", 101, "title")
    try:
        dispatcher.dispatch("not_a_real_tool", {})
        assert False, "expected ValueError"
    except ValueError:
        pass
