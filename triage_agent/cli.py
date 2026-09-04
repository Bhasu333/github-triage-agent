"""CLI entrypoint.

Usage:
    export GITHUB_TOKEN=...      # required: avoids the 60/hr unauth rate limit
    export ANTHROPIC_API_KEY=... # required: for classify/draft steps
    python -m triage_agent.cli --repo expressjs/express --limit 5
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from .embeddings import Embedder
from .github_client import GitHubClient
from .pipeline import TriagePipeline
from .state import TriageState
from .vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous GitHub issue triage agent")
    parser.add_argument("--repo", required=True, help="owner/repo, e.g. expressjs/express")
    parser.add_argument("--limit", type=int, default=10, help="max open issues to pull")
    parser.add_argument("--state-file", default="state.json")
    parser.add_argument("--chroma-path", default="chroma_store", help="vector store directory")
    parser.add_argument("--no-semantic", action="store_true", help="disable RAG retrieval, use keyword overlap only")
    parser.add_argument("--mcp", action="store_true", help="run tool execution over the real MCP protocol (Phase 3)")
    args = parser.parse_args()

    try:
        owner, repo = args.repo.split("/", 1)
    except ValueError:
        print("--repo must be in owner/repo format", file=sys.stderr)
        sys.exit(1)

    github = GitHubClient()
    if args.no_semantic:
        state = TriageState(args.state_file)
    else:
        state = TriageState(args.state_file, vector_store=VectorStore(args.chroma_path), embedder=Embedder())

    if args.mcp:
        # MCP server subprocess reads state via env vars, not constructor args.
        import os
        os.environ["TRIAGE_STATE_FILE"] = args.state_file
        os.environ["TRIAGE_CHROMA_PATH"] = args.chroma_path
        from .pipeline_mcp import TriagePipelineMCP
        pipeline = TriagePipelineMCP(github, state)
        results = asyncio.run(pipeline.run(owner, repo, limit=args.limit))
    else:
        pipeline = TriagePipeline(github, state)
        results = pipeline.run(owner, repo, limit=args.limit)

    if not results:
        print("No new issues to triage (all fetched issues already in state, or none open).")
        return

    print(f"\nTriaged {len(results)} issue(s) in {args.repo}:\n")
    for r in results:
        print(f"  #{r['issue']:<6} [{r['priority']}/{r['type']:<15}] conf={r['confidence']:.2f}  {r['title'][:60]}")
    print(f"\nFull records written to {args.state_file}")


if __name__ == "__main__":
    main()
