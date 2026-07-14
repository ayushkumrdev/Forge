"""Web fetch tool: lets the agent read documentation and reference pages from
the internet. Read-only (no permission needed), returns text with HTML stripped
to keep it cheap for a local model's context window."""

from __future__ import annotations

import re
from typing import Any

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


def _html_to_text(html: str) -> str:
    html = _SCRIPT_STYLE_RE.sub(" ", html)
    html = _TAG_RE.sub(" ", html)
    for entity, char in _ENTITIES.items():
        html = html.replace(entity, char)
    html = _WS_RE.sub(" ", html)
    html = _BLANKLINES_RE.sub("\n\n", html)
    return "\n".join(line.strip() for line in html.splitlines()).strip()
