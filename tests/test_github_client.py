import json
from pathlib import Path

from triage_agent.github_client import Issue

FIXTURE = Path(__file__).parent.parent / "fixtures" / "sample_issue.json"


def test_issue_from_api_parses_fixture():
    data = json.loads(FIXTURE.read_text())
    issue = Issue.from_api(data)
    assert issue.number == 101
    assert issue.author == "demo-reporter"
    assert "needs-triage" in issue.labels
    assert issue.comments_count == 2


def test_issue_from_api_handles_null_body():
    data = json.loads(FIXTURE.read_text())
    data["body"] = None
    issue = Issue.from_api(data)
    assert issue.body == ""
