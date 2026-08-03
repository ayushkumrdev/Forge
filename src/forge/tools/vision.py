"""Vision tool — Forge's eyes. A local multimodal model (llava, minicpm-v,
qwen2.5vl, …) describes or extracts content from image files so the text-only
coder model can work with screenshots, UI mockups, diagrams and error photos.

Read-only; the image never leaves the machine."""

from __future__ import annotations

import base64
from typing import Any

import httpx

from forge.safety.guard import SafetyGuard
from forge.tools.base import Tool, ToolResult

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

_DEFAULT_QUESTION = (
    "Describe this image in detail for a software engineer. Transcribe any "
    "text exactly (code, error messages, labels, log lines). Describe UI "
    "elements, layout, diagrams, and anything else relevant."
)
_MAX_IMAGE_BYTES = 20 * 1024 * 1024


class ReadImageTool(Tool):
    name = "read_image"
    mutating = False
    description = (
        "Look at an image file (screenshot, UI mockup, diagram, photo of an "
        "error) and get a detailed text description, with any visible text "
        "transcribed. Optionally ask a specific question about the image."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Image path relative to the repository."},
            "question": {
                "type": "string",
                "description": "Optional: a specific question about the image.",
            },
        },
        "required": ["path"],
    }

    def __init__(
        self,
        guard: SafetyGuard,
        host: str = "http://localhost:11434",
        model: str = "llava:7b",
        timeout_s: float = 180.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._guard = guard
        self._host = host
        self.model = model
        self._timeout_s = timeout_s
        self._transport = transport

    def run(self, path: str, question: str = "") -> ToolResult:
        resolved = self._guard.resolve_path(path)
        if not resolved.is_file():
            return ToolResult(ok=False, error=f"Image not found: {path}")
        if resolved.suffix.lower() not in IMAGE_EXTENSIONS:
            return ToolResult(
                ok=False,
                error=f"{path} is not an image "
                f"(supported: {', '.join(sorted(IMAGE_EXTENSIONS))}).",
            )
        raw = resolved.read_bytes()
        if len(raw) > _MAX_IMAGE_BYTES:
            return ToolResult(ok=False, error=f"Image too large ({len(raw)} bytes).")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": question or _DEFAULT_QUESTION,
                    "images": [base64.b64encode(raw).decode("ascii")],
                }
            ],
            "stream": False,
        }
        try:
            with httpx.Client(
                base_url=self._host, timeout=self._timeout_s, transport=self._transport
            ) as client:
                response = client.post("/api/chat", json=payload)
                response.raise_for_status()
        except httpx.ConnectError:
            return ToolResult(
                ok=False,
                error="Cannot reach Ollama for vision. Start it with `ollama serve`.",
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return ToolResult(
                    ok=False,
                    error=f"Vision model {self.model!r} is not pulled. "
                    f"Run: ollama pull {self.model}",
                )
            return ToolResult(
                ok=False, error=f"Vision request failed: HTTP {exc.response.status_code}"
            )
        except httpx.HTTPError as exc:
            return ToolResult(ok=False, error=f"Vision request failed: {exc}")

        content = ((response.json().get("message") or {}).get("content") or "").strip()
        if not content:
            return ToolResult(ok=False, error="The vision model returned no description.")
        return ToolResult(ok=True, output=f"[image {path} seen by {self.model}]\n{content}")
