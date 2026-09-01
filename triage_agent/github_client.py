"""Thin wrapper around the GitHub REST API for fetching issues and comments."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import requests

GITHUB_API = "https://api.github.com"


class GitHubAPIError(RuntimeError):
    pass


@dataclass
class Issue:
    number: int
    title: str
    body: str
    labels: list[str]
    author: str
    created_at: str
    comments_count: int
    html_url: str
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "Issue":
        return cls(
            number=data["number"],
            title=data["title"],
            body=data.get("body") or "",
            labels=[l["name"] for l in data.get("labels", [])],
            author=data["user"]["login"],
            created_at=data["created_at"],
            comments_count=data.get("comments", 0),
            html_url=data["html_url"],
            raw=data,
        )


class GitHubClient:
    """Read-only client. Works unauthenticated (60 req/hr) or with a token
    (5000 req/hr) via the GITHUB_TOKEN env var."""

    def __init__(self, token: str | None = None, session: requests.Session | None = None):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.session = session or requests.Session()
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self.session.headers.update(headers)

    def _get(self, path: str, params: dict | None = None) -> Any:
        resp = self.session.get(f"{GITHUB_API}{path}", params=params or {})
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            raise GitHubAPIError(
                "GitHub rate limit exceeded. Set GITHUB_TOKEN for a 5000 req/hr limit."
            )
        if not resp.ok:
            raise GitHubAPIError(f"GitHub API error {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def get_open_issues(self, owner: str, repo: str, limit: int = 20) -> list[Issue]:
        """Fetch open issues, excluding pull requests (GitHub's issues
        endpoint returns both; PRs carry a 'pull_request' key)."""
        issues: list[Issue] = []
        page = 1
        while len(issues) < limit:
            batch = self._get(
                f"/repos/{owner}/{repo}/issues",
                params={"state": "open", "per_page": min(100, limit), "page": page},
            )
            if not batch:
                break
            for item in batch:
                if "pull_request" in item:
                    continue
                issues.append(Issue.from_api(item))
                if len(issues) >= limit:
                    break
            page += 1
        return issues

    def get_issue(self, owner: str, repo: str, number: int) -> Issue:
        return Issue.from_api(self._get(f"/repos/{owner}/{repo}/issues/{number}"))

    def get_issue_comments(self, owner: str, repo: str, number: int, limit: int = 10) -> list[str]:
        data = self._get(f"/repos/{owner}/{repo}/issues/{number}/comments", params={"per_page": limit})
        return [c["body"] for c in data]
