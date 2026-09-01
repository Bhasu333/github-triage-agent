"""Tool definitions (JSON schemas for Claude) and the dispatcher that
executes them against GitHubClient / TriageState.

Three tools:
  - get_issue_details: pull full issue body + comment thread
  - get_similar_past_triages: pull relevant triage history/context
  - submit_triage: REQUIRED terminal tool call; structured output contract
"""
from __future__ import annotations

from typing import Any

from .github_client import GitHubClient
from .state import TriageState

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_issue_details",
        "description": "Fetch the full body and recent comments for the current issue being triaged.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_similar_past_triages",
        "description": (
            "Retrieve similar previously-triaged issues and aggregate priority/label "
            "history for this repo, to keep classification decisions consistent as the "
            "backlog grows."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "submit_triage",
        "description": (
            "REQUIRED final step. Submit the structured triage decision for this issue. "
            "You must call this exactly once, after gathering enough context, to finish."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "priority": {"type": "string", "enum": ["P0", "P1", "P2", "P3"]},
                "issue_type": {
                    "type": "string",
                    "enum": ["bug", "feature_request", "question", "duplicate", "documentation"],
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "labels_suggested": {"type": "array", "items": {"type": "string"}},
                "draft_response": {
                    "type": "string",
                    "description": "A first-pass reply to post to the issue, for maintainer review.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Why this priority/type, referencing issue content and any similar past triages.",
                },
            },
            "required": ["priority", "issue_type", "confidence", "labels_suggested", "draft_response", "reasoning"],
        },
    },
]


class ToolDispatcher:
    """Binds tool calls to a specific (owner, repo, issue) context."""

    def __init__(self, github: GitHubClient, state: TriageState, owner: str, repo: str, issue_number: int, issue_title: str):
        self.github = github
        self.state = state
        self.owner = owner
        self.repo = repo
        self.issue_number = issue_number
        self.issue_title = issue_title
        self.repo_key = f"{owner}/{repo}"

    def dispatch(self, tool_name: str, tool_input: dict) -> str:
        if tool_name == "get_issue_details":
            return self._get_issue_details()
        if tool_name == "get_similar_past_triages":
            return self._get_similar_past_triages()
        if tool_name == "submit_triage":
            # Terminal tool: the agent loop reads tool_input directly to build
            # a TriageRecord. Dispatcher just acknowledges receipt.
            return "Triage decision received."
        raise ValueError(f"Unknown tool: {tool_name}")

    def _get_issue_details(self) -> str:
        issue = self.github.get_issue(self.owner, self.repo, self.issue_number)
        comments = self.github.get_issue_comments(self.owner, self.repo, self.issue_number, limit=5)
        parts = [
            f"Title: {issue.title}",
            f"Author: {issue.author}",
            f"Existing labels: {issue.labels}",
            f"Body:\n{issue.body[:3000]}",
        ]
        if comments:
            parts.append("Recent comments:\n" + "\n---\n".join(c[:500] for c in comments))
        return "\n\n".join(parts)

    def _get_similar_past_triages(self) -> str:
        summary = self.state.context_summary(self.repo_key)
        # find_similar_semantic uses vector retrieval when a vector store is
        # configured on the state; otherwise it transparently falls back to
        # keyword overlap. Callers don't need to know which one ran.
        similar = self.state.find_similar_semantic(self.repo_key, self.issue_title)
        if not similar:
            return summary + "\n\nNo similar past-triaged issues found."
        lines = [summary, "\nMost similar past triages (retrieved by embedding similarity):"]
        for r in similar:
            lines.append(f"  #{r['issue_number']} '{r['title']}' -> {r['priority']}/{r['issue_type']}: {r['reasoning'][:150]}")
        return "\n".join(lines)
