"""GitHub intelligence tools — read-only access to any GitHub repository.

`github_repo` gives the agent a one-shot architectural picture of a repo
(metadata, README, full file tree); `github_file` reads any file from it.
To MODIFY a GitHub project the agent clones it into the workspace with git
and edits locally through the normal safe tool set — these tools never write.

Unauthenticated GitHub API allows 60 requests/hour; set FORGE_GITHUB_TOKEN
for 5000/hour and private-repo access."""

from __future__ import annotations

import re
from typing import Any

import httpx

from forge.tools.base import Tool, ToolResult

_API = "https://api.github.com"
_MAX_README_CHARS = 6_000
_MAX_TREE_ENTRIES = 300
_MAX_FILE_CHARS = 100_000
_TIMEOUT_S = 30.0

_URL_RE = re.compile(r"github\.com[/:]([^/\s]+)/([^/\s#?]+)")
_SHORT_RE = re.compile(r"^([\w.-]+)/([\w.-]+)$")


def parse_repo(spec: str) -> tuple[str, str] | None:
    """Accepts 'owner/repo', a github.com URL (https or ssh), with or without
    .git / trailing paths. Returns (owner, repo) or None."""
    spec = spec.strip().rstrip("/")
    match = _URL_RE.search(spec) or _SHORT_RE.fullmatch(spec)
    if not match:
        return None
    owner, repo = match.group(1), match.group(2)
    return owner, repo.removesuffix(".git")


def _client(token: str, transport: httpx.BaseTransport | None) -> httpx.Client:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "forge-agent",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(base_url=_API, headers=headers, timeout=_TIMEOUT_S, transport=transport)


def _friendly_error(response: httpx.Response, what: str) -> str:
    if response.status_code == 404:
        return f"{what} not found on GitHub — check the owner/repo (and path)."
    if response.status_code in (403, 429):
        if response.headers.get("X-RateLimit-Remaining") == "0":
            return (
                "GitHub API rate limit reached (60/hour unauthenticated). "
                "Set FORGE_GITHUB_TOKEN for 5000/hour, or wait and retry."
            )
        return (
            f"GitHub refused the request (HTTP {response.status_code}). "
            "Private repo? Set FORGE_GITHUB_TOKEN."
        )
    return f"GitHub returned HTTP {response.status_code}: {response.text[:300]}"


class GitHubRepoTool(Tool):
    name = "github_repo"
    description = (
        "Analyze a GitHub repository without cloning it: metadata, README, "
        "and the full file tree — the fastest way to understand a project's "
        "architecture. Accepts 'owner/repo' or a github.com URL. Read-only. "
        "To MODIFY the project, git clone it into the workspace afterwards."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository as 'owner/repo' or a github.com URL.",
            },
        },
        "required": ["repo"],
    }

    def __init__(self, token: str = "", transport: httpx.BaseTransport | None = None) -> None:
        self._token = token
        self._transport = transport

    def run(self, repo: str) -> ToolResult:
        parsed = parse_repo(repo)
        if parsed is None:
            return ToolResult(
                ok=False,
                error=f"Cannot parse {repo!r} — use 'owner/repo' or a github.com URL.",
            )
        owner, name = parsed
        with _client(self._token, self._transport) as client:
            meta_resp = client.get(f"/repos/{owner}/{name}")
            if meta_resp.status_code != 200:
                return ToolResult(
                    ok=False, error=_friendly_error(meta_resp, f"Repository {owner}/{name}")
                )
            meta = meta_resp.json()

            sections = [
                f"# {meta.get('full_name', f'{owner}/{name}')}",
                f"{meta.get('description') or '(no description)'}",
                f"language: {meta.get('language') or 'n/a'} · "
                f"stars: {meta.get('stargazers_count', 0)} · "
                f"forks: {meta.get('forks_count', 0)} · "
                f"default branch: {meta.get('default_branch', 'main')} · "
                f"license: {(meta.get('license') or {}).get('spdx_id') or 'none'}",
            ]
            if meta.get("topics"):
                sections.append("topics: " + ", ".join(meta["topics"]))
            clone_url = meta.get("clone_url", f"https://github.com/{owner}/{name}.git")
            sections.append(f"clone: git clone {clone_url}")

            readme_resp = client.get(
                f"/repos/{owner}/{name}/readme",
                headers={"Accept": "application/vnd.github.raw+json"},
            )
            if readme_resp.status_code == 200:
                readme = readme_resp.text
                if len(readme) > _MAX_README_CHARS:
                    readme = readme[:_MAX_README_CHARS] + "\n... [README truncated]"
                sections.append("## README\n" + readme)

            branch = meta.get("default_branch", "main")
            tree_resp = client.get(f"/repos/{owner}/{name}/git/trees/{branch}?recursive=1")
            if tree_resp.status_code == 200:
                data = tree_resp.json()
                entries = [
                    item["path"] + ("/" if item.get("type") == "tree" else "")
                    for item in data.get("tree", [])
                ]
                shown = entries[:_MAX_TREE_ENTRIES]
                note = (
                    f"\n... [{len(entries) - len(shown)} more entries]"
                    if len(entries) > len(shown)
                    else ""
                )
                sections.append(
                    f"## File tree ({len(entries)} entries)\n" + "\n".join(shown) + note
                )

        return ToolResult(ok=True, output="\n\n".join(sections))


class GitHubFileTool(Tool):
    name = "github_file"
    description = (
        "Read one file from a GitHub repository (no clone needed). Use after "
        "github_repo to inspect specific source files. Read-only."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository as 'owner/repo' or a github.com URL.",
            },
            "path": {"type": "string", "description": "File path inside the repository."},
            "ref": {
                "type": "string",
                "description": "Branch, tag or commit (default: the default branch).",
            },
        },
        "required": ["repo", "path"],
    }

    def __init__(self, token: str = "", transport: httpx.BaseTransport | None = None) -> None:
        self._token = token
        self._transport = transport

    def run(self, repo: str, path: str, ref: str = "") -> ToolResult:
        parsed = parse_repo(repo)
        if parsed is None:
            return ToolResult(
                ok=False,
                error=f"Cannot parse {repo!r} — use 'owner/repo' or a github.com URL.",
            )
        owner, name = parsed
        params = {"ref": ref} if ref else None
        with _client(self._token, self._transport) as client:
            response = client.get(
                f"/repos/{owner}/{name}/contents/{path.lstrip('/')}",
                params=params,
                headers={"Accept": "application/vnd.github.raw+json"},
            )
        if response.status_code != 200:
            return ToolResult(
                ok=False, error=_friendly_error(response, f"{owner}/{name}:{path}")
            )
        text = response.text
        if len(text) > _MAX_FILE_CHARS:
            text = text[:_MAX_FILE_CHARS] + "\n... [file truncated]"
        at = f" @ {ref}" if ref else ""
        return ToolResult(ok=True, output=f"[{owner}/{name}:{path}{at}]\n{text}")
