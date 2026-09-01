"""Persistent state store for triage history, per repo.

This is what makes the agent's decisions consistent across a growing
backlog instead of re-deriving priority from scratch on every issue:
each triage call can pull recent history + label/priority frequency
as context before classifying the next issue.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .embeddings import Embedder
from .vector_store import VectorStore


@dataclass
class TriageRecord:
    issue_number: int
    title: str
    priority: str
    issue_type: str
    confidence: float
    labels_suggested: list[str]
    draft_response: str
    reasoning: str
    triaged_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TriageState:
    def __init__(
        self,
        path: str | Path = "state.json",
        vector_store: VectorStore | None = None,
        embedder: Embedder | None = None,
    ):
        self.path = Path(path)
        self._data: dict = self._load()
        # Both optional: state.json history/frequency stats always work standalone.
        # Semantic retrieval only activates when both are supplied (see pipeline.py).
        self.vector_store = vector_store
        self.embedder = embedder

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2))

    def _repo_bucket(self, repo_key: str) -> dict:
        return self._data.setdefault(
            repo_key, {"triaged": [], "label_frequency": {}, "priority_frequency": {}}
        )

    def already_triaged(self, repo_key: str, issue_number: int) -> bool:
        bucket = self._repo_bucket(repo_key)
        return any(r["issue_number"] == issue_number for r in bucket["triaged"])

    def record(self, repo_key: str, record: TriageRecord, index_text: str | None = None) -> None:
        bucket = self._repo_bucket(repo_key)
        bucket["triaged"].append(asdict(record))
        bucket["priority_frequency"][record.priority] = (
            bucket["priority_frequency"].get(record.priority, 0) + 1
        )
        for label in record.labels_suggested:
            bucket["label_frequency"][label] = bucket["label_frequency"].get(label, 0) + 1
        self._save()

        if self.vector_store and self.embedder:
            text = index_text or record.title
            embedding = self.embedder.embed(text)
            self.vector_store.upsert(
                repo_key,
                record.issue_number,
                embedding,
                document=text,
                metadata={
                    "title": record.title,
                    "priority": record.priority,
                    "issue_type": record.issue_type,
                    "reasoning": record.reasoning[:300],
                },
            )

    def context_summary(self, repo_key: str, max_recent: int = 5) -> str:
        """A compact text summary handed to the agent before it classifies
        the next issue — this is the 'priority context carried across
        issues' claim made real."""
        bucket = self._repo_bucket(repo_key)
        if not bucket["triaged"]:
            return "No prior triage history for this repo yet."
        recent = bucket["triaged"][-max_recent:]
        lines = [
            f"Triaged so far: {len(bucket['triaged'])} issues.",
            f"Priority distribution: {bucket['priority_frequency']}",
            f"Common labels: {bucket['label_frequency']}",
            "Most recent triage decisions:",
        ]
        for r in recent:
            lines.append(f"  #{r['issue_number']} '{r['title'][:50]}' -> {r['priority']}/{r['issue_type']}")
        return "\n".join(lines)

    def find_similar(self, repo_key: str, title: str, top_k: int = 3) -> list[dict]:
        """Cheap keyword-overlap similarity over past triaged titles.
        No embeddings in v1 — see PRD Phase 2 for the vector-search upgrade."""
        bucket = self._repo_bucket(repo_key)
        query_words = set(title.lower().split())
        scored = []
        for r in bucket["triaged"]:
            past_words = set(r["title"].lower().split())
            overlap = len(query_words & past_words)
            if overlap > 0:
                scored.append((overlap, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]

    def find_similar_semantic(self, repo_key: str, query_text: str, top_k: int = 3) -> list[dict]:
        """Vector-similarity retrieval over indexed issue text. Falls back
        to the keyword method if no vector store/embedder was configured."""
        if not (self.vector_store and self.embedder):
            return self.find_similar(repo_key, query_text, top_k)
        embedding = self.embedder.embed(query_text)
        hits = self.vector_store.query(repo_key, embedding, top_k)
        return [
            {
                "issue_number": h["issue_number"],
                "title": h["metadata"]["title"],
                "priority": h["metadata"]["priority"],
                "issue_type": h["metadata"]["issue_type"],
                "reasoning": h["metadata"]["reasoning"],
                "distance": h["distance"],
            }
            for h in hits
        ]
