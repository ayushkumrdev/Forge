"""Web tools: search the internet and read pages. Read-only (no permission
needed); results are stripped to text to keep them cheap for a local model's
context window."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import unquote

import httpx

from forge.tools.base import Tool, ToolResult

_MAX_CHARS = 12_000
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")
_BLANKLINES_RE = re.compile(r"\n\s*\n\s*\n+")

_ENTITIES = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'",
    "&nbsp;": " ", "&mdash;": "—", "&ndash;": "–", "&hellip;": "…",
}


class FetchUrlTool(Tool):
    name = "fetch_url"
    # read-only: fetching a public page has no side effects, so no approval
    mutating = False
    description = (
        "Fetch a web page (documentation, API reference, README) and return its "
        "text content with HTML stripped. Use this to look up unfamiliar libraries "
        "or APIs. Only http/https URLs."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The http(s) URL to fetch."},
        },
        "required": ["url"],
    }

    def __init__(
        self, timeout_s: float = 20.0, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._timeout_s = timeout_s
        self._transport = transport

    def run(self, url: str) -> ToolResult:
        if not url.lower().startswith(("http://", "https://")):
            return ToolResult(ok=False, error="Only http:// and https:// URLs are allowed.")
        try:
            with httpx.Client(
                timeout=self._timeout_s, follow_redirects=True, transport=self._transport
            ) as client:
                response = client.get(url, headers={"User-Agent": "Forge/1.0 (local agent)"})
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return ToolResult(ok=False, error=f"HTTP {exc.response.status_code} fetching {url}")
        except httpx.HTTPError as exc:
            return ToolResult(ok=False, error=f"Could not fetch {url}: {exc}")

        content_type = response.headers.get("content-type", "")
        body = response.text
        text = body if "html" not in content_type.lower() else _html_to_text(body)
        truncated = len(text) > _MAX_CHARS
        if truncated:
            text = text[:_MAX_CHARS]
        header = f"[fetched {url} — {len(body)} bytes{' , truncated' if truncated else ''}]\n"
        return ToolResult(ok=True, output=header + text)


_SEARCH_URL = "https://html.duckduckgo.com/html/"
_RESULT_LINK_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL
)
_RESULT_SNIPPET_RE = re.compile(
    r'class="result__snippet"[^>]*>(.*?)</(?:a|td|div)>', re.DOTALL
)


def _decode_ddg_href(href: str) -> str:
    """DuckDuckGo links go through a redirect with the real URL in ?uddg=."""
    if "uddg=" in href:
        return unquote(href.split("uddg=", 1)[1].split("&", 1)[0])
    if href.startswith("//"):
        return "https:" + href
    return href


class WebSearchTool(Tool):
    name = "web_search"
    # read-only: searching has no side effects, so no approval needed
    mutating = False
    description = (
        "Search the web (DuckDuckGo) and get the top results: title, URL and "
        "snippet. Use for unfamiliar errors, libraries, or current information "
        "— then fetch_url the most promising result to read it."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "max_results": {
                "type": "integer",
                "description": "Number of results to return (default 5, max 10).",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self, timeout_s: float = 20.0, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._timeout_s = timeout_s
        self._transport = transport

    def run(self, query: str, max_results: int = 5) -> ToolResult:
        limit = max(1, min(max_results, 10))
        try:
            with httpx.Client(
                timeout=self._timeout_s, follow_redirects=True, transport=self._transport
            ) as client:
                response = client.get(
                    _SEARCH_URL,
                    params={"q": query},
                    headers={"User-Agent": "Mozilla/5.0 (Forge local agent)"},
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                ok=False, error=f"Search failed: HTTP {exc.response.status_code}"
            )
        except httpx.HTTPError as exc:
            return ToolResult(ok=False, error=f"Search failed: {exc}")

        links = _RESULT_LINK_RE.findall(response.text)[:limit]
        snippets = [
            _html_to_text(s) for s in _RESULT_SNIPPET_RE.findall(response.text)[:limit]
        ]
        if not links:
            return ToolResult(
                ok=True, output=f"No results for {query!r}. Try different terms."
            )
        lines = []
        for i, (href, title_html) in enumerate(links):
            snippet = snippets[i] if i < len(snippets) else ""
            lines.append(
                f"{i + 1}. {_html_to_text(title_html)}\n   {_decode_ddg_href(href)}"
                + (f"\n   {snippet}" if snippet else "")
            )
        return ToolResult(
            ok=True,
            output=f"Top {len(lines)} results for {query!r}:\n" + "\n".join(lines),
        )


_NUM_ENTITY_RE = re.compile(r"&#(x?)([0-9a-fA-F]{1,6});")


def _decode_numeric_entity(match: re.Match[str]) -> str:
    try:
        return chr(int(match.group(2), 16 if match.group(1) else 10))
    except (ValueError, OverflowError):
        return match.group(0)


def _html_to_text(html: str) -> str:
    html = _SCRIPT_STYLE_RE.sub(" ", html)
    html = _TAG_RE.sub(" ", html)
    for entity, char in _ENTITIES.items():
        html = html.replace(entity, char)
    html = _NUM_ENTITY_RE.sub(_decode_numeric_entity, html)
    html = _WS_RE.sub(" ", html)
    html = _BLANKLINES_RE.sub("\n\n", html)
    return "\n".join(line.strip() for line in html.splitlines()).strip()
