"""Async pipeline variant for the MCP-based agent (each issue triage
opens/closes its own MCP session -- simple and correct, though a
longer-lived session pool would be the natural next optimization)."""
from __future__ import annotations

from .github_client import GitHubClient
from .mcp_agent import MCPTriageAgent
from .state import TriageState


class TriagePipelineMCP:
    def __init__(self, github: GitHubClient, state: TriageState, agent: MCPTriageAgent | None = None):
        self.github = github
        self.state = state
        self.agent = agent or MCPTriageAgent()

    async def run(self, owner: str, repo: str, limit: int = 10) -> list[dict]:
        repo_key = f"{owner}/{repo}"
        issues = self.github.get_open_issues(owner, repo, limit=limit)
        results = []
        for issue in issues:
            if self.state.already_triaged(repo_key, issue.number):
                continue
            record = await self.agent.triage(owner, repo, issue)
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
