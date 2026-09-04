# GitHub Issue Triage Agent

Autonomous triage agent using Claude's tool-calling API. Pulls open issues from a
GitHub repo, classifies priority/type, drafts a first-pass reply, and persists
state so later decisions stay consistent with earlier ones as the backlog grows.

See `PRD.md` for the full design doc.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your real GITHUB_TOKEN and ANTHROPIC_API_KEY
export $(cat .env | xargs)
```

- `GITHUB_TOKEN`: free, read-only fine-grained PAT. Without it you're capped at
  60 requests/hr unauthenticated, which this project's earlier dev log ran into.
- `ANTHROPIC_API_KEY`: from console.anthropic.com. Real cost per run — small,
  but not zero.

## Run

```bash
python -m triage_agent.cli --repo expressjs/express --limit 5
```

Output is a CLI summary plus a `state.json` with full structured records
(priority, type, confidence, draft response, reasoning) per issue.

## Test (no keys required)

```bash
PYTHONPATH=. python -m pytest tests/ -v
```

## Architecture

```
CLI -> TriagePipeline -> TriageAgent (Claude tool-calling loop)
                              |-- get_issue_details      -> GitHubClient
                              |-- get_similar_past_triages -> TriageState
                              `-- submit_triage (required terminal tool call)
```

## Status

Core pipeline + RAG retrieval layer built and unit-tested offline (17/17
passing). Live end-to-end run against a real repo done with real credentials.

## RAG layer (Phase 2)

`get_similar_past_triages` retrieves by embedding similarity (Chroma, one
collection per repo) instead of keyword overlap. Embeddings are produced by
scikit-learn's `HashingVectorizer` — deterministic, no pretrained-model
download required. Swapping in a neural embedding model later (OpenAI/Cohere/
local sentence-transformers) is a one-class change: `Embedder.embed()` is the
only interface `VectorStore` and `TriageState` depend on.

Disable it with `--no-semantic` to fall back to the original keyword-overlap
matcher.

## MCP protocol layer (Phase 3)

`mcp_server.py` exposes the three tools (`get_issue_details`,
`get_similar_past_triages`, `submit_triage`) as a real MCP server over stdio.
`mcp_agent.py` is an MCP *client*: it discovers tool schemas at runtime via
`list_tools()` (not hardcoded), hands them to Claude, and executes every tool
call as a genuine MCP request/response across a process boundary — the same
protocol Sycamore's stack uses for tool execution.

Run with it:
```bash
python -m triage_agent.cli --repo expressjs/express --limit 3 --mcp
```

Without `--mcp`, the original Phase 1 in-process `ToolDispatcher` (`tools.py`,
`agent.py`) still runs — both paths are live and tested, so the diff between
"in-process function call" and "protocol-based tool execution" is something
you can point to directly, not just describe.

Tests (`test_mcp_server.py`) spawn the real server as a subprocess and talk to
it over actual stdio MCP — not mocked — using `pytest-asyncio`
(`pip install pytest-asyncio` for local test runs).

## Roadmap

- Longer-lived MCP session (pool instead of one session per issue).
- Real neural embeddings once a model-download path is available (see
  `embeddings.py` docstring — the interface is already set up for a
  drop-in swap).
