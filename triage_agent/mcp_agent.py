"""Phase 3: the same multi-step tool-calling loop as agent.py, but tool
execution now goes over the real MCP protocol (stdio) to a separate
server process (mcp_server.py) instead of an in-process function call.

Tool schemas are discovered at runtime via session.list_tools() -- not
hardcoded here -- and handed directly to Claude as its tool definitions,
since MCP tool schemas and Claude's tool schemas are both JSON Schema.
"""
from __future__ import annotations

import os

import anthropic
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .github_client import Issue
from .state import TriageRecord

MODEL = "claude-sonnet-4-6"
MAX_TURNS = 6

SYSTEM_PROMPT = """You are a GitHub issue triage assistant. For the given issue, \
you must: (1) understand what it's reporting, (2) check it against past triage \
history for consistency, (3) decide priority and type, (4) draft a helpful \
first-pass reply, and (5) call submit_triage exactly once with your structured \
decision. Use get_issue_details and get_similar_past_triages as needed before \
deciding -- do not guess without reading the issue body. Tools that operate on \
a specific repo take owner/repo as explicit arguments; use the owner/repo given \
in the task."""


def _mcp_tool_to_claude_schema(tool) -> dict:
    return {"name": tool.name, "description": tool.description or "", "input_schema": tool.inputSchema}


class MCPTriageAgent:
    def __init__(self, server_command: list[str] | None = None, client: anthropic.Anthropic | None = None):
        self.client = client or anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        cmd = server_command or ["python3", "-m", "triage_agent.mcp_server"]
        self.server_params = StdioServerParameters(command=cmd[0], args=cmd[1:])

    async def triage(self, owner: str, repo: str, issue: Issue) -> TriageRecord:
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                claude_tools = [_mcp_tool_to_claude_schema(t) for t in tools_result.tools]

                messages = [
                    {
                        "role": "user",
                        "content": (
                            f"Triage issue #{issue.number} in {owner}/{repo}: '{issue.title}'\n"
                            f"Existing labels: {issue.labels}\n"
                            "Fetch full details and check triage history before deciding."
                        ),
                    }
                ]

                for _ in range(MAX_TURNS):
                    response = self.client.messages.create(
                        model=MODEL, max_tokens=1500, system=SYSTEM_PROMPT,
                        tools=claude_tools, messages=messages,
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
                        args = dict(tool_use.input)
                        # Fill in repo context the model may omit -- it's implied
                        # by the task, not something it should have to restate.
                        if tool_use.name == "get_issue_details":
                            args.setdefault("owner", owner)
                            args.setdefault("repo", repo)
                            args.setdefault("issue_number", issue.number)
                        elif tool_use.name == "get_similar_past_triages":
                            args.setdefault("owner", owner)
                            args.setdefault("repo", repo)
                            args.setdefault("issue_title", issue.title)

                        result = await session.call_tool(tool_use.name, args)
                        text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
                        tool_results.append({"type": "tool_result", "tool_use_id": tool_use.id, "content": text})
                    messages.append({"role": "user", "content": tool_results})

                raise RuntimeError(f"Agent hit MAX_TURNS ({MAX_TURNS}) without submitting triage on issue #{issue.number}")
