"""These tests spawn the actual mcp_server.py as a subprocess and talk to
it over real stdio MCP protocol -- not a mock. They don't touch GitHub or
Claude (only get_similar_past_triages and submit_triage are exercised,
neither of which needs network), so they run without any API keys."""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

PROJECT_ROOT = Path(__file__).parent.parent


def _server_params(state_dir: Path) -> StdioServerParameters:
    env = dict(os.environ)
    env["TRIAGE_STATE_FILE"] = str(state_dir / "state.json")
    env["TRIAGE_CHROMA_PATH"] = str(state_dir / "chroma")
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "triage_agent.mcp_server"],
        cwd=str(PROJECT_ROOT),
        env=env,
    )


@pytest.mark.asyncio
async def test_list_tools_returns_all_three_registered_tools():
    with tempfile.TemporaryDirectory() as tmp:
        params = _server_params(Path(tmp))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = {t.name for t in tools.tools}
                assert names == {"get_issue_details", "get_similar_past_triages", "submit_triage"}


@pytest.mark.asyncio
async def test_submit_triage_round_trip_over_real_protocol():
    with tempfile.TemporaryDirectory() as tmp:
        params = _server_params(Path(tmp))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "submit_triage",
                    {
                        "priority": "P1",
                        "issue_type": "bug",
                        "confidence": 0.9,
                        "labels_suggested": ["bug"],
                        "draft_response": "Thanks for the report.",
                        "reasoning": "Reproducible crash.",
                    },
                )
                text = "".join(c.text for c in result.content if hasattr(c, "text"))
                assert "received" in text.lower()


@pytest.mark.asyncio
async def test_get_similar_past_triages_reads_seeded_state():
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp)
        # Seed state.json directly, exactly as the pipeline would after a
        # prior run, so the MCP server reads real prior history.
        state_file = state_dir / "state.json"
        state_file.write_text(json.dumps({
            "acme/widgets": {
                "triaged": [{
                    "issue_number": 5, "title": "Crash on empty input", "priority": "P0",
                    "issue_type": "bug", "confidence": 0.95, "labels_suggested": ["bug", "crash"],
                    "draft_response": "...", "reasoning": "Null pointer on empty input.",
                    "triaged_at": "2026-08-01T00:00:00Z",
                }],
                "label_frequency": {"bug": 1, "crash": 1},
                "priority_frequency": {"P0": 1},
            }
        }))

        params = _server_params(state_dir)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "get_similar_past_triages",
                    {"owner": "acme", "repo": "widgets", "issue_title": "App crashes on empty input"},
                )
                text = "".join(c.text for c in result.content if hasattr(c, "text"))
                assert "#5" in text
                assert "Triaged so far: 1" in text
