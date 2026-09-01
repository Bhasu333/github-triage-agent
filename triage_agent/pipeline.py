"""Orchestrates the fetch -> classify -> draft -> review pipeline across
a repo's open-issue backlog, skipping issues already triaged in state."""
from __future__ import annotations

from .agent import TriageAgent
from .github_client import GitHubClient
from .state import TriageState


class TriagePipeline:
    def __init__(self, github: GitHubClient, state: TriageState, agent: TriageAgent | None = None):
        self.github = github
        self.state = state
        self.agent = agent or TriageAgent()

    def run(self, owner: str, repo: str, limit: int = 10) -> list[dict]:
        repo_key = f"{owner}/{repo}"
        issues = self.github.get_open_issues(owner, repo, limit=limit)
        results = []
        for issue in issues:
            if self.state.already_triaged(repo_key, issue.number):
                continue
            record = self.agent.triage(self.github, self.state, owner, repo, issue)
            # Index title+body (not just the classification output) so semantic
            # retrieval matches on what the issue is actually about.
            index_text = f"{issue.title}\n{issue.body}"
            self.state.record(repo_key, record, index_text=index_text)
            results.append(
                {
                    "issue": issue.number,
                    "title": issue.title,
                    "priority": record.priority,
                    "type": record.issue_type,
                    "confidence": record.confidence,
                }
            )
        return results
