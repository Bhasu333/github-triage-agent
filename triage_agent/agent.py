"""The actual multi-step tool-calling agent, per issue.

Loop: Claude receives the issue summary + tool schemas, decides which
tools to call (details, history), and must terminate by calling
submit_triage. Max turns enforced so a stuck loop fails loudly instead
of hanging or burning API credits silently.
"""
from __future__ import annotations

import os

import anthropic

from .github_client import GitHubClient, Issue
from .state import TriageRecord, TriageState
from .tools import TOOL_SCHEMAS, ToolDispatcher

MODEL = "claude-sonnet-4-6"
MAX_TURNS = 6

SYSTEM_PROMPT = """You are a GitHub issue triage assistant. For the given issue, \
you must: (1) understand what it's reporting, (2) check it against past triage \
history for consistency, (3) decide priority and type, (4) draft a helpful \
first-pass reply, and (5) call submit_triage exactly once with your structured \
decision. Use get_issue_details and get_similar_past_triages as needed before \
deciding — do not guess without reading the issue body."""


class TriageAgent:
    def __init__(self, client: anthropic.Anthropic | None = None):
        self.client = client or anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def triage(self, github: GitHubClient, state: TriageState, owner: str, repo: str, issue: Issue) -> TriageRecord:
        dispatcher = ToolDispatcher(github, state, owner, repo, issue.number, issue.title)
        messages = [
            {
                "role": "user",
                "content": (
                    f"Triage issue #{issue.number}: '{issue.title}'\n"
                    f"Existing labels: {issue.labels}\n"
                    "Fetch full details and check triage history before deciding."
                ),
            }
        ]

        for turn in range(MAX_TURNS):
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                raise RuntimeError(f"Agent stopped without calling submit_triage on issue #{issue.number}")

            submit_call = next((t for t in tool_uses if t.name == "submit_triage"), None)
            if submit_call:
                return TriageRecord(
                    issue_number=issue.number,
                    title=issue.title,
                    priority=submit_call.input["priority"],
                    issue_type=submit_call.input["issue_type"],
                    confidence=submit_call.input["confidence"],
                    labels_suggested=submit_call.input["labels_suggested"],
                    draft_response=submit_call.input["draft_response"],
                    reasoning=submit_call.input["reasoning"],
                )

            tool_results = []
            for tool_use in tool_uses:
                result_text = dispatcher.dispatch(tool_use.name, tool_use.input)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": tool_use.id, "content": result_text}
                )
            messages.append({"role": "user", "content": tool_results})

        raise RuntimeError(f"Agent hit MAX_TURNS ({MAX_TURNS}) without submitting triage on issue #{issue.number}")
