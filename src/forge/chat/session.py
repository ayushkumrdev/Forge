"""Interactive chat session — the Claude Code experience on a local model.

One conversation, persistent across turns (and across restarts via the
transcript file), where the model works directly in the repository through
the full tool set. Mutating tools go through the permission policy, every
file change is ledgered for /undo and /diff, @file mentions inline file
content, and long conversations are compacted with an LLM summary."""

from __future__ import annotations

import json
import re
from pathlib import Path

from forge.agents.base import recover_inline_tool_call
from forge.chat.instructions import load_project_instructions
from forge.config import ForgeSettings
from forge.llm.base import ChatMessage, LLMClient, Usage
from forge.memory.store import MemoryStore
from forge.safety.guard import SafetyGuard
from forge.safety.permissions import PermissionPolicy
from forge.telemetry import Recorder
from forge.tools.base import ToolRegistry
from forge.tools.changes import ChangeLedger
from forge.tools.code_intel import FindSymbolTool, WhoImportsTool
from forge.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from forge.tools.git_tool import GitTool
from forge.tools.retrieval_tool import SearchCodeTool
from forge.tools.search import GlobTool, GrepTool
from forge.tools.terminal import RunCommandTool

CHAT_SYSTEM = """You are Forge, an autonomous AI software engineer with DIRECT \
tool access to the user's repository, running locally — like Claude Code.

CRITICAL — you act through tools, not through the user:
- NEVER ask the user to paste file contents: read files yourself (read_file).
- NEVER show code and tell the user to apply it: edit the files yourself
  (edit_file / write_file).
- NEVER tell the user to run a command: run it yourself (run_command).
- To call a tool, reply with ONLY a JSON object and nothing else:
  {"name": "<tool_name>", "arguments": {"<param>": "<value>"}}
  One tool call per reply; you will get the result back, then continue.
- Only reply in plain text once the request is fully handled (or to answer a
  question that needs no repository access).

Working rules:
- ALWAYS read a file before editing it. Prefer edit_file for small changes;
  write_file for new files or full rewrites you have just read.
- edit_file old_string must be copied EXACTLY from read_file output. If an
  edit fails twice on a file, read it and use write_file with the complete
  corrected content instead.
- When your new code uses a symbol, make sure that file imports or defines it.
- Use find_symbol / who_imports / search_code / grep to locate code. Never
  guess paths or invent APIs.
- Verify changes when possible (run tests, or python -m py_compile).
- If a tool call is denied by the user, do not retry it; ask what to do."""

_TREE_IN_PROMPT_CHARS = 2_500

_MENTION_RE = re.compile(r"@([\w\-./\\]+\.[\w]+)")
_KEEP_RECENT_ON_COMPACT = 8
_MAX_MENTION_CHARS = 6_000
# qwen occasionally leaks chat-template tokens into long tool conversations
_SPECIAL_TOKEN_RE = re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>")


class ChatSession:
    def __init__(
        self,
        workspace: Path,
        llm: LLMClient,
        settings: ForgeSettings | None = None,
        policy: PermissionPolicy | None = None,
        recorder: Recorder | None = None,
        store: MemoryStore | None = None,
        session_id: str = "chat",
    ) -> None:
        self.workspace = workspace.resolve()
        self.llm = llm
        self.settings = settings or ForgeSettings()
        self.session_id = session_id
        self.recorder = recorder or Recorder(session_id, self.workspace, store=store)
        self.usage = Usage()
        self.history: list[ChatMessage] = []

        self._guard = SafetyGuard(self.workspace)
        self.ledger = ChangeLedger(self.workspace, session_id)
        self.registry = self._build_registry(policy)
        tree = self.snapshot.tree
        if len(tree) > _TREE_IN_PROMPT_CHARS:
            tree = tree[:_TREE_IN_PROMPT_CHARS] + "\n... [tree truncated]"
        self._system = (
            CHAT_SYSTEM
            + f"\n\n## Repository ({self.workspace})\n{tree}"
        )
        instructions = load_project_instructions(self.workspace)
        if instructions:
            self._system += "\n\n" + instructions

    def _build_registry(self, policy: PermissionPolicy | None) -> ToolRegistry:
        from forge.repo.scanner import RepoScanner
        from forge.retrieval.engine import RetrievalEngine

        snapshot = RepoScanner(
            self.workspace,
            max_tree_entries=self.settings.max_tree_entries,
            cache_path=self.workspace / ".forge" / "repo_index.json",
        ).scan()
        self.snapshot = snapshot
        engine = RetrievalEngine(self.workspace)
        engine.build(snapshot)
        return ToolRegistry(
            [
                ReadFileTool(self._guard),
                WriteFileTool(self._guard, self.ledger),
                EditFileTool(self._guard, self.ledger),
                ListDirTool(self._guard),
                RunCommandTool(self._guard, self.workspace, self.settings.command_timeout_s),
                GrepTool(self.workspace),
                GlobTool(self.workspace),
                GitTool(self.workspace),
                FindSymbolTool(snapshot),
                WhoImportsTool(snapshot),
                SearchCodeTool(engine),
            ],
            policy=policy,
        )

    # -- conversation ------------------------------------------------------------

    def send(self, user_text: str) -> str:
        """One user turn: run the tool loop until the model answers in text."""
        self.history.append(
            ChatMessage(role="user", content=self._expand_mentions(user_text))
        )
        self._maybe_compact()

        nudged = False
        for _step in range(self.settings.max_agent_steps):
            response = self.llm.chat(
                [ChatMessage(role="system", content=self._system), *self.history],
                tools=self.registry.specs(),
            )
            self.usage.add(response.usage)
            message = recover_inline_tool_call(response.message, self.registry.names())

            if not message.tool_calls:
                content = _SPECIAL_TOKEN_RE.sub("", message.content).strip()
                if not content and not nudged:
                    # empty/garbage reply (template-token leak): one retry nudge
                    nudged = True
                    self.history.append(
                        ChatMessage(
                            role="user",
                            content="Your last reply was empty. Continue with the "
                            "request, or summarize what you did.",
                        )
                    )
                    continue
                self.history.append(message.model_copy(update={"content": content}))
                self.save_transcript()
                return content

            self.history.append(message)
            for call in message.tool_calls:
                self.recorder.event("chat", "tool_call", tool=call.name, arguments=call.arguments)
                result = self.registry.execute(call.name, call.arguments)
                self.recorder.event(
                    "chat", "tool_result", tool=call.name, ok=result.ok, error=result.error
                )
                self.history.append(
                    ChatMessage(
                        role="tool",
                        content=result.render(self.settings.max_tool_output_chars),
                        tool_name=call.name,
                    )
                )

        self.save_transcript()
        note = "Stopped: step budget exhausted for this turn."
        self.history.append(ChatMessage(role="assistant", content=note))
        return note

    def _expand_mentions(self, text: str) -> str:
        """Inline @path/to/file.ext mentions, Claude Code style."""
        blocks: list[str] = []
        for mention in dict.fromkeys(_MENTION_RE.findall(text)):
            try:
                resolved = self._guard.resolve_path(mention.replace("\\", "/"))
            except Exception:  # noqa: BLE001 — bad mention is just text
                continue
            if resolved.is_file():
                content = resolved.read_text(encoding="utf-8-sig", errors="replace")
                if len(content) > _MAX_MENTION_CHARS:
                    content = content[:_MAX_MENTION_CHARS] + "\n... [truncated]"
                blocks.append(f"--- content of {mention} ---\n{content}")
        if blocks:
            text += "\n\n" + "\n\n".join(blocks)
        return text

    # -- compaction ----------------------------------------------------------------

    def _maybe_compact(self, force: bool = False) -> bool:
        threshold = self.settings.chat_compact_threshold
        if not force and len(self.history) <= threshold:
            return False
        old = self.history[:-_KEEP_RECENT_ON_COMPACT]
        recent = self.history[-_KEEP_RECENT_ON_COMPACT:]
        if not old:
            return False
        digest = "\n".join(
            f"[{m.role}{'/' + m.tool_name if m.tool_name else ''}] {m.content[:400]}"
            for m in old
            if m.content
        )
        response = self.llm.chat(
            [
                ChatMessage(
                    role="system",
                    content="Summarize this conversation between a user and a coding "
                    "agent in under 250 words. Keep: the user's goals, decisions "
                    "made, files changed, and any unresolved problems. Plain text.",
                ),
                ChatMessage(role="user", content=digest),
            ]
        )
        self.usage.add(response.usage)
        summary = ChatMessage(
            role="system",
            content="Summary of the earlier conversation (older messages were "
            "compacted):\n" + response.message.content,
        )
        self.history = [summary, *recent]
        self.recorder.event("chat", "compacted", kept=len(recent))
        return True

    def compact(self) -> bool:
        return self._maybe_compact(force=True)

    # -- persistence -----------------------------------------------------------------

    @property
    def transcript_path(self) -> Path:
        return self.workspace / ".forge" / "chat" / f"{self.session_id}.json"

    def save_transcript(self) -> None:
        try:
            self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
            self.transcript_path.write_text(
                json.dumps([m.model_dump() for m in self.history], indent=1),
                encoding="utf-8",
            )
        except OSError:  # transcripts are a convenience, never a failure
            pass

    def load_transcript(self) -> bool:
        if not self.transcript_path.exists():
            return False
        try:
            data = json.loads(self.transcript_path.read_text(encoding="utf-8"))
            self.history = [ChatMessage.model_validate(m) for m in data]
            return True
        except (OSError, json.JSONDecodeError, ValueError):
            return False

    # -- session actions ----------------------------------------------------------------

    def undo(self) -> list[str]:
        """Revert every file this session changed to its original state."""
        restored = self.ledger.restore_all()
        if restored:
            self.history.append(
                ChatMessage(
                    role="system",
                    content="The user reverted all file changes from this session: "
                    + ", ".join(restored),
                )
            )
            self.save_transcript()
        return restored

    def clear(self) -> None:
        self.history = []
        self.save_transcript()
