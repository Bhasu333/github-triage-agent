"""MCP server exposing the triage tools over the real Model Context
Protocol, stdio transport.

Phase 2's tools.py dispatched tool calls as plain in-process Python
function calls. This module moves tool *execution* across a real
process boundary: the agent (mcp_agent.py) runs as an MCP client, this
runs as a separate MCP server process, and every tool call is an
actual MCP request/response over stdio -- discoverable via
list_tools(), not hardcoded in the client.

Because a server can serve many concurrent clients/issues, tools here
take (owner, repo, issue_number/title) as explicit arguments rather
than being bound to one issue's context at construction time (the
Phase 1 ToolDispatcher pattern).

Run standalone for manual testing:
    python -m triage_agent.mcp_server
"""
from __future__ import annotations

import os

from mcp.server.mcpserver import MCPServer

from .embeddings import Embedder
from .github_client import GitHubClient
from .state import TriageState
from .vector_store import VectorStore

STATE_FILE = os.environ.get("TRIAGE_STATE_FILE", "state.json")
CHROMA_PATH = os.environ.get("TRIAGE_CHROMA_PATH", "chroma_store")

_github = GitHubClient()
_state = TriageState(STATE_FILE, vector_store=VectorStore(CHROMA_PATH), embedder=Embedder())

server = MCPServer("github-triage-tools")


@server.tool()
def get_issue_details(owner: str, repo: str, issue_number: int) -> str:
    """Fetch the full body and recent comments for a GitHub issue."""
    issue = _github.get_issue(owner, repo, issue_number)
    comments = _github.get_issue_comments(owner, repo, issue_number, limit=5)
    parts = [
        f"Title: {issue.title}",
        f"Author: {issue.author}",
        f"Existing labels: {issue.labels}",
        f"Body:\n{issue.body[:3000]}",
    ]
    if comments:
        parts.append("Recent comments:\n" + "\n---\n".join(c[:500] for c in comments))
    return "\n\n".join(parts)


@server.tool()
def get_similar_past_triages(owner: str, repo: str, issue_title: str) -> str:
    """Retrieve similar previously-triaged issues (via embedding
    retrieval) and aggregate priority/label history for this repo."""
    repo_key = f"{owner}/{repo}"
    summary = _state.context_summary(repo_key)
    similar = _state.find_similar_semantic(repo_key, issue_title)
    if not similar:
        return summary + "\n\nNo similar past-triaged issues found."
    lines = [summary, "\nMost similar past triages (retrieved by embedding similarity):"]
    for r in similar:
        lines.append(f"  #{r['issue_number']} '{r['title']}' -> {r['priority']}/{r['issue_type']}: {r['reasoning'][:150]}")
    return "\n".join(lines)


@server.tool()
def submit_triage(
    priority: str,
    issue_type: str,
    confidence: float,
    labels_suggested: list[str],
    draft_response: str,
    reasoning: str,
) -> str:
    """REQUIRED final step. Submit the structured triage decision for
    the issue currently being triaged."""
    return "Triage decision received."


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
