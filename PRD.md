# GitHub Issue Triage Agent — PRD & Technical Plan

## 1. Problem
Maintainers of active repos accumulate open issues faster than they can read them. First-pass triage — is this a bug, a duplicate, urgent, or noise — is repetitive and low-leverage work that a tool-calling agent can do reliably if it can (a) read the issue, (b) check it against triage history, and (c) produce a structured, reviewable output instead of a black-box guess.

## 2. Goal
Build an autonomous agent that pulls open issues from a GitHub repo, classifies priority/type, drafts a first-pass response, and logs its reasoning — with state that persists across runs so later decisions are informed by earlier ones.

**Out of scope (v1):** auto-posting comments/labels to GitHub (human-in-the-loop review only), multi-repo support, fine-tuning a classifier.

## 3. Users
You (as the maintainer/reviewer) running it against your own or a demo repo. Output is a reviewable report, not autonomous write access — intentional, since granting write access to a public repo from an agent is a real trust boundary, not just a demo detail.

## 4. Functional Requirements
1. **Fetch** — pull N open issues from a given `owner/repo`, excluding PRs.
2. **Classify** — for each issue, determine priority (P0–P3), type (bug/feature/question/duplicate), and confidence.
3. **Draft** — generate a first-pass reply grounded in the issue content.
4. **Persist** — store every triage decision (issue #, classification, draft, timestamp) in local state.
5. **Adapt** — before classifying, retrieve similar past-triaged issues from state and pass them as context, so decisions are consistent across a growing backlog (this is the "persistent state-tracking loop" already claimed on the resume — v1 makes it real).
6. **Report** — output a human-readable summary (CLI table + JSON) for review.

## 5. Non-Functional Requirements
- Must run against a real public repo with a real GitHub token (free, read-only scope).
- Must degrade gracefully on GitHub rate limits (clear error, not a silent hang).
- Every Claude API call must be a genuine multi-step tool-calling loop (fetch details → search history → submit structured output), not one-shot prompting — this is the actual technical claim, so it has to be true.
- Testable offline via recorded fixtures (no live API needed to verify logic).

## 6. Architecture

```
                 ┌────────────┐
   CLI ─────────▶│  Pipeline  │
                 └─────┬──────┘
                       │ per issue
                       ▼
              ┌─────────────────┐
              │  TriageAgent     │◀──── Claude API (tool-calling loop)
              │  (agent.py)      │
              └───┬─────────┬────┘
                  │         │
        tools.py  │         │  tools.py
    get_issue_details   get_similar_past_triages
                  │         │
                  ▼         ▼
          GitHubClient   TriageState (JSON store)
          (github_client.py)  (state.py)
```

Loop per issue: Claude decides which tools to call (issue details, comment thread, similar past triages), then must call `submit_triage` with structured JSON (priority, type, confidence, draft_response, reasoning) to terminate. That structured tool call is the enforced output contract — no free-text parsing.

## 7. Data Model (state.json)
```json
{
  "owner/repo": {
    "triaged": [
      {
        "issue_number": 42,
        "priority": "P1",
        "type": "bug",
        "confidence": 0.82,
        "labels_suggested": ["bug", "regression"],
        "draft_response": "...",
        "reasoning": "...",
        "triaged_at": "2026-09-02T18:00:00Z"
      }
    ],
    "label_frequency": {"bug": 12, "question": 4},
    "priority_frequency": {"P0": 1, "P1": 6, "P2": 9}
  }
}
```
`label_frequency` / `priority_frequency` are what `get_similar_past_triages` reasons over — cheap, explainable, no vector DB needed for v1.

## 8. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12 | Matches resume, matches Claude SDK maturity |
| LLM | Claude API (`anthropic` SDK), tool use | Multi-step tool-calling is the actual feature being demonstrated |
| GitHub access | `requests` against GitHub REST v3 | No need for a heavier SDK (PyGithub) for this scope |
| State store | Local JSON file (v1) → SQLite (v2) | JSON is honest about scope; don't reach for a DB you don't need yet |
| CLI | `argparse` | No framework needed for a single-command tool |
| Testing | `pytest` + recorded fixtures | Verify tool-calling logic without live API dependency |
| Secrets | `.env` + `python-dotenv`, never committed | `GITHUB_TOKEN`, `ANTHROPIC_API_KEY` |
| Packaging | `requirements.txt`, optional `pyproject.toml` | Keep it simple for a personal project repo |

## 9. What You Need Before Building
- **GitHub personal access token**, read-only, no scopes needed for public repos (Settings → Developer settings → Fine-grained tokens). Unauthenticated requests are capped at 60/hr and will hit rate limits fast.
- **Anthropic API key** (console.anthropic.com) — this is a real cost center, even if small; track usage.
- Both go in a local `.env` file, loaded via `python-dotenv`, never pasted into chat or committed to git.

## 10. Milestones
1. **Core pipeline** — `github_client.py` (done), `state.py`, `tools.py`, `agent.py`, `pipeline.py`, `cli.py`. Runs end-to-end against a real repo with real keys.
2. **Tests** — fixture-based tests for state adaptation logic and tool dispatch, no live network required.
3. **README + demo run** — recorded output against a real public repo, committed to GitHub with real timestamps.
4. **Phase 2 (TWG-relevant): RAG layer** — DONE. `embeddings.py` (HashingVectorizer, deterministic, no model download) + `vector_store.py` (Chroma, per-repo collections) replace keyword-overlap similarity with real embedding retrieval over full issue text. `TriageState.find_similar_semantic` falls back to keyword matching if no vector store is configured, so the core pipeline still runs standalone. 8 new tests verify semantic ordering and repo isolation.
5. **Phase 3 (Sycamore-relevant): MCP rebuild** — DONE. `mcp_server.py` exposes the three tools over real MCP (stdio transport); `mcp_agent.py`/`pipeline_mcp.py` run the client side, discovering tool schemas at runtime instead of hardcoding them. `--mcp` flag on the CLI switches transport. 3 new tests spawn the actual server subprocess and exercise real protocol round trips (tool discovery, submit_triage, and a seeded-state retrieval), all without requiring API keys. Original in-process dispatch path (Phase 1) still works unmodified, so both are live and comparable.

## 11. Success Criteria
- Runs against a real repo end-to-end with no fabricated output.
- State visibly changes agent behavior on the 2nd+ run of a session (provable via test).
- Every resume bullet describing this project maps to a line of code you can point to and explain in an interview.

## 12. Risks
- GitHub rate limits without a token — mitigated by requiring one up front.
- Claude tool-calling loop not terminating — mitigated by max-turn cap and required `submit_triage` call.
- Scope creep into RAG/MCP before v1 is solid — sequence matters; ship the core loop first.
