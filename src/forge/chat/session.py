"""Interactive chat session — a fully autonomous engineer on a local model.

One conversation, persistent across turns (and across restarts via the
transcript file), where the model works directly in the repository through
the full tool set. Mutating tools go through the permission policy, every
file change is ledgered for /undo and /diff, @file mentions inline file
content, and long conversations are compacted with an LLM summary."""

from __future__ import annotations

import json
import os
import platform
import re
import sys
import time
from pathlib import Path

from forge.agents.base import constrained_tool_retry, recover_inline_tool_call
from forge.chat.formatting import normalize_markdown
from forge.chat.instructions import load_project_instructions
from forge.config import ForgeSettings
from forge.llm.base import ChatMessage, LLMClient, Usage
from forge.llm.json_utils import looks_like_tool_call
from forge.memory.store import MemoryStore
from forge.safety.guard import SafetyGuard
from forge.safety.permissions import PermissionPolicy
from forge.telemetry import Recorder
from forge.tools.base import ToolRegistry
from forge.tools.changes import ChangeLedger
from forge.tools.code_intel import FindSymbolTool, WhoImportsTool
from forge.tools.filesystem import (
    AppendFileTool,
    DeleteFileTool,
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from forge.tools.git_tool import GitTool
from forge.tools.github import GitHubFileTool, GitHubRepoTool
from forge.tools.retrieval_tool import SearchCodeTool
from forge.tools.search import GlobTool, GrepTool
from forge.tools.terminal import PowerShellTool, RunCommandTool
from forge.tools.vision import IMAGE_EXTENSIONS, ReadImageTool
from forge.tools.web import FetchUrlTool, WebSearchTool
from forge.verify.coverage import (
    Requirement,
    assess,
    build_evidence,
    coverage_nudge,
    decompose,
    focused_prompt,
    looks_multi_requirement,
)
from forge.verify.ladder import Ladder
from forge.verify.resolution import (
    dangling_reference_errors,
    undefined_self_call_errors,
)

CHAT_SYSTEM = """You are Forge, an elite autonomous AI software engineer running \
locally on the user's machine with FULL tool access to their repository. You do \
the work yourself — the user talks, you engineer.

## Tool protocol — follow exactly
To use a tool, reply with ONLY one JSON object, nothing else:
{"name": "<tool>", "arguments": {"<param>": "<value>"}}
After each result you are called again; chain as many tool steps as needed.
Reply in plain text ONLY when the request is fully handled (or when it needs
no repository access at all).

Example — read, then edit:
{"name": "read_file", "arguments": {"path": "app.py"}}
...result arrives, then...
{"name": "edit_file", "arguments": {"path": "app.py", "old_string": "x=1", "new_string": "x=2"}}

## Prime directives
1. ACT yourself. NEVER ask the user to paste code, apply a change, or run a
   command — you have tools for all of it.
2. Words don't change files; tool calls do. NEVER paste code into the chat
   for the user to apply, and NEVER say "I will now edit X" as your reply —
   if code belongs in a file, put it there with edit_file/write_file THIS
   step. A reply that shows code without having written it is a failure.
3. Truth comes from tools, never from memory. Read files before talking about
   them. Never invent files, functions, or APIs — verify with find_symbol or
   grep first.
4. Finish the WHOLE request. Before your final answer, re-read the request
   and confirm every part is done.
5. Verify your work: run the project's tests, or at least
   `python -m py_compile <file>` after changing Python. Never declare success
   while checks fail.
6. Keep changes minimal and in the repository's existing style — match its
   indentation, naming, imports, and patterns.

## Workflow for every coding task
1. UNDERSTAND — identify the target files (find_symbol / search_code / grep /
   list_dir). 2. READ — read_file every file you will touch; never edit
   unread files. 3. CHANGE — edit_file for surgical changes (old_string
   copied EXACTLY from the read output, with enough lines to be unique);
   append_to_file to ADD a new function/class to an existing file;
   write_file for new files or full rewrites. 4. VERIFY — run_command the
   tests or a compile check and read the output. 5. REPORT — plain-text
   summary: what changed, which files, how you verified it.

## Recovery playbook
- edit_file "not found": re-read the file and copy the exact text, including
  whitespace and blank lines.
- Adding something new to a file? Use append_to_file with just the new code.
  Never call edit_file with an empty old_string — it cannot append.
- edit_file failed twice on one file: read it, then write_file the COMPLETE
  corrected content.
- Your new code uses a symbol: that file must import or define it — tests too.
- A command fails: read the error, fix the root cause, run again. Never repeat
  a failing command unchanged.
- A tool call is denied by the user: do not retry it; ask how to proceed.
- Unfamiliar API or error: grep the codebase for existing usage, or
  web_search it and fetch_url the best result.

## Tools available
read_file · write_file · append_to_file · edit_file · delete_file ·
list_dir · run_command ·
run_powershell (full PowerShell on Windows: file ops, processes, env,
package managers) · grep · find_files · search_code (by meaning) ·
find_symbol (find definitions) · who_imports (what depends on a file) ·
git (status/diff/log/add/commit) · web_search (search the web) ·
fetch_url (read a web page) ·
github_repo (analyze any GitHub repository: metadata, README, file tree) ·
github_file (read one file from a GitHub repository) ·
read_image (see an image file — screenshot, mockup, diagram — described in
detail with text transcribed; available when a vision model is configured)

## GitHub workflow
- Asked about a GitHub project? github_repo first (architecture + README),
  then github_file for the specific sources. Never guess what a repo contains.
- Asked to MODIFY a GitHub project? Analyze with github_repo, then clone it
  INTO the workspace: run_command `git clone <url> <folder>` — then work on
  the cloned files with the normal read/edit/write tools and verify.

## How to write your final answer
Lead with the outcome in one sentence — what you did, or the direct answer.
Details come after, for the reader who wants them.

Format for a skimmer, using real Markdown:
- Put a BLANK LINE between paragraphs, before a list, and after a list.
  Without blank lines everything runs together into one wall of text.
- Use `- ` bullets for parallel points and `1. ` for ordered steps. One idea
  per bullet, one line each where possible.
- Use `**bold**` for the few words that matter, `inline code` for file names,
  paths, commands, symbols and values — never write a bare filename.
- Use a ``` fenced block WITH a language tag for any code or command output.
  Never indent code with spaces instead of fencing it.
- Use `## ` headings only when the answer has genuinely separate sections.
  A three-sentence answer needs no headings.

Length matches the question: a factual question gets one or two sentences,
not a report. Do not pad with restatements of the request, apologies, or
"let me know if you need anything else". No filler.

When you changed code, close with a short summary of what changed, in which
files, and how you verified it."""

_TREE_IN_PROMPT_CHARS = 2_500

_MENTION_RE = re.compile(r"@([\w\-./\\]+\.[\w]+)")
_KEEP_RECENT_ON_COMPACT = 8
_MAX_MENTION_CHARS = 6_000
# qwen occasionally leaks chat-template tokens into long tool conversations
_SPECIAL_TOKEN_RE = re.compile(
    r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|</?tool_response>|</?tool_call>"
)

# -- act-don't-tell enforcement ---------------------------------------------------
# The classic small-model deflection: asked to change code, it pastes the code
# into chat (or promises to do it) instead of calling tools. The gate refuses
# to accept such a reply as final while nothing was actually changed.

# Verb STEMS plus an optional inflection, so "make", "makes", "making" and
# "changing" all arm the gate — an earlier word-list version missed every
# inflected form and silently under-armed on requests like "make it return 0".
_ACTION_REQUEST_RE = re.compile(
    r"\b(?:add|fix|writ|creat|implement|refactor|chang|updat|remov|delet|"
    r"renam|mov|install|build|convert|replac|improv|optimi[sz]|correct|"
    r"patch|appl|extract|split|merg|migrat|upgrad|mak|ensur|handl|support|"
    r"guard|wire|enabl|disabl|rework|rewrit|set up|clean up)"
    r"(?:e|es|ed|ing|s)?\b",
    re.IGNORECASE,
)
# A question about code is not a request to change it: "how do I add …?"
# must not arm the gate, or explanatory answers get bounced as deflection.
_QUESTION_RE = re.compile(
    r"^\s*(?:how|what|what's|why|when|where|which|who|can|could|would|should|"
    r"does|do|did|is|are|was|explain|tell me|describe)\b",
    re.IGNORECASE,
)


def is_action_request(text: str) -> bool:
    """True when the user is asking for a change rather than an explanation."""
    if _ACTION_REQUEST_RE.search(text) is None:
        return False
    return not (_QUESTION_RE.match(text) and text.rstrip().endswith("?"))
_CODE_FENCE_RE = re.compile(r"```[\w+-]*\n.+?```", re.DOTALL)
_PROMISE_RE = re.compile(
    r"\b(?:i\s*(?:'ll|will)|let\s+me|i\s*(?:'m|am)\s+going\s+to|"
    r"i\s+can\s+now|next,?\s+i)\b[^.!\n]{0,80}?"
    r"\b(?:writ|edit|creat|add|fix|updat|chang|implement|appl|refactor|run|mak)",
    re.IGNORECASE,
)
_MAX_ACTION_NUDGES = 2
_MAX_COVERAGE_PASSES = 2

# "Did it act?" means "did it change the repository". run_command is a
# mutating tool because it needs permission to touch system state, but a turn
# that only ran a command has not done the work — counting it as action
# inflated ADT and stopped the act-don't-tell gate from firing on a model
# that just re-ran the tests instead of writing code.
_FILE_MUTATING_TOOLS = frozenset(
    {"write_file", "edit_file", "append_to_file", "delete_file"}
)

_GENIUS_CHECK = (
    "Final completeness check: re-read the user's ORIGINAL request at the "
    "start of this turn. If ANY part of it is not done or not verified, do "
    "it NOW with tools. If everything is complete and verified, restate "
    "your final summary."
)

# system prompt for the optional second model (the "thinker"): it never
# touches tools — it turns the user's raw message into a precise brief the
# coder model can act on without misreading intent
_THINKER_SYSTEM = """You are the reasoning half of a two-model AI software \
engineer. A separate coder model will act on the user's message with real \
tools. Your job: read the user's message and write a short brief so the coder \
cannot misunderstand it.

State plainly:
1. INTENT — what the user actually wants, in one sentence.
2. STEPS — the concrete actions to take, in order.
3. WHERE — likely files/areas involved (best guess from the message).
4. VERIFY — how the coder should prove it worked.

Under 150 words. Plain text only, no code, no markdown headers. If the \
message is a simple question needing no file changes, reply with just: \
QUESTION: <the question restated precisely>."""

_PASTED_CODE_NUDGE = (
    "You pasted code into the chat instead of applying it. The user asked you "
    "to make this change — you have tools, so make it yourself NOW: call "
    "edit_file or write_file with that exact code, then verify with "
    "run_command. Do not reply in plain text until the change is actually in "
    "the repository files."
)
_PROMISE_NUDGE = (
    "You described what you would do instead of doing it. Execute it NOW with "
    "tool calls (read_file, edit_file, write_file, run_command). Do not "
    "narrate the plan; perform it, then report what changed."
)

# claiming verification that never happened is the purest hallucination —
# catch replies that say tests/checks ran when no command ran this turn
_FALSE_VERIFICATION_RE = re.compile(
    r"\b(?:ran|run(?:ning)?)\b[^.\n]{0,50}\b(?:tests?|checks?|pytest|linter)\b"
    r"|\b(?:tests?|checks?)\s+(?:pass(?:ed)?|succeed(?:ed)?)\b"
    r"|\bno\s+errors?\s+(?:found|reported)\b"
    r"|\bverified\b[^.\n]{0,50}\b(?:running|tests?|command)\b",
    re.IGNORECASE,
)
# ...but a reply that DISCLAIMS verification ("I did not run the tests",
# "please run the tests yourself") is honest, and must never be scored as a
# false claim — the disclaimer is precisely the behaviour the gate wants.
_DISCLAIMER_RE = re.compile(
    r"\b(?:not|n't|never|unable|cannot|can't|please|recommend|suggest|should|"
    r"you\s+(?:can|could|may|must|need|will|might)|if\s+you|try\s+running|"
    r"feel\s+free|make\s+sure\s+to)\b",
    re.IGNORECASE,
)


def claims_verification(content: str) -> bool:
    """True only when the reply asserts that IT verified something. Sentences
    that deny or defer verification do not count, so an honest 'I did not run
    the tests' is never mistaken for a lie — by the gate or by the metrics."""
    for sentence in re.split(r"(?<=[.!?\n])\s+", content):
        if _FALSE_VERIFICATION_RE.search(sentence) and not _DISCLAIMER_RE.search(sentence):
            return True
    return False
_FALSE_CLAIM_NUDGE = (
    "You claimed the tests/checks ran, but you did NOT run any command this "
    "turn. Run them NOW with run_command and report the actual output — or "
    "state plainly that you did not run them. Never claim verification you "
    "have not performed."
)


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
        thinker_llm: LLMClient | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.llm = llm
        self.settings = settings or ForgeSettings()
        self.session_id = session_id
        self.recorder = recorder or Recorder(session_id, self.workspace, store=store)
        self.usage = Usage()
        self.history: list[ChatMessage] = []
        # optional hooks set by hosts (the desktop app wires these):
        self.on_stream = None  # Callable[[str], None] — receives content deltas
        self.on_step = None  # Callable[[], None] — called before each LLM step
        self.should_stop = None  # Callable[[], bool] — soft cancel

        self._thinker = thinker_llm
        if self._thinker is None and self.settings.thinker_model:
            from forge.llm.factory import make_client

            self._thinker = make_client(self.settings, model=self.settings.thinker_model)
        self.effort = (
            self.settings.effort
            if self.settings.effort in ("fast", "smart", "genius")
            else "smart"
        )

        self._guard = SafetyGuard(self.workspace)
        self.ledger = ChangeLedger(self.workspace, session_id)
        self.registry = self._build_registry(policy)
        tree = self.snapshot.tree
        if len(tree) > _TREE_IN_PROMPT_CHARS:
            tree = tree[:_TREE_IN_PROMPT_CHARS] + "\n... [tree truncated]"
        shells = "run_command (cmd.exe) and run_powershell" if os.name == "nt" else "run_command"
        self._system = (
            CHAT_SYSTEM
            + f"\n\n## Environment\nOS: {platform.system()} {platform.release()} · "
            f"Python {sys.version.split()[0]} · shell tools: {shells}. "
            "Use the right syntax for this OS."
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
        self._engine = engine
        ladder = Ladder(
            self.workspace,
            resolution=self.settings.gate_resolution,
            types=self.settings.gate_types,
        )
        tools = [
            ReadFileTool(self._guard),
            WriteFileTool(
                self._guard, self.ledger, self.settings.syntax_gate, ladder=ladder
            ),
            AppendFileTool(
                self._guard, self.ledger, self.settings.syntax_gate, ladder=ladder
            ),
            EditFileTool(
                self._guard,
                self.ledger,
                self.settings.syntax_gate,
                self.settings.gate_edit_repair,
                ladder=ladder,
            ),
            DeleteFileTool(self._guard, self.ledger),
            ListDirTool(self._guard),
            RunCommandTool(self._guard, self.workspace, self.settings.command_timeout_s),
            GrepTool(self.workspace),
            GlobTool(self.workspace),
            GitTool(self.workspace),
            FindSymbolTool(snapshot),
            WhoImportsTool(snapshot),
            SearchCodeTool(engine),
            FetchUrlTool(),
            WebSearchTool(),
            GitHubRepoTool(self.settings.github_token),
            GitHubFileTool(self.settings.github_token),
        ]
        if os.name == "nt":
            tools.append(
                PowerShellTool(self._guard, self.workspace, self.settings.command_timeout_s)
            )
        self._vision: ReadImageTool | None = None
        if self.settings.vision_model:
            self._vision = ReadImageTool(
                self._guard, host=self.settings.ollama_host, model=self.settings.vision_model
            )
            tools.append(self._vision)
        return ToolRegistry(tools, policy=policy)

    # -- conversation ------------------------------------------------------------

    def send(self, user_text: str) -> str:
        """One user turn: run the tool loop until the model answers in text."""
        expanded = self._expand_mentions(user_text)
        if (
            expanded == user_text
            and self.effort != "fast"
            and self.settings.gate_preflight
        ):
            # no @file context supplied -> pre-flight retrieval puts the real
            # relevant code in front of the model before it generates
            context = self._engine.preflight(user_text)
            if context:
                expanded += "\n\n" + context
        brief = self._interpret(user_text)
        if brief:
            expanded += (
                "\n\n## Intent brief (a reasoning model interpreted the request"
                " — follow it unless the user's own words contradict it)\n" + brief
            )
        self.history.append(ChatMessage(role="user", content=expanded))
        self._maybe_compact()

        # arm the act-don't-tell gate when this turn asks for a change
        action_turn = is_action_request(user_text)
        turn_mutated = False
        turn_ran_command = False
        corrections = 0
        nudged = False
        genius_checked = False
        coverage_passes = 0
        dangling_checked = False
        requirements: list[Requirement] | None = None
        commands_run: list[str] = []
        # the evaluation harness reconstructs per-turn behaviour from the trace
        self.recorder.event(
            "chat", "turn_started", action_turn=action_turn, effort=self.effort
        )
        deadline = (
            time.monotonic() + self.settings.max_turn_seconds
            if self.settings.max_turn_seconds > 0
            else None
        )
        for _step in range(self._step_budget()):
            if self._cancelled():
                return self._finish_stopped()
            if deadline is not None and time.monotonic() > deadline:
                self.recorder.event("chat", "turn_timeout", output=str(self.ledger.changed_files))
                return self._finish_turn(
                    "Stopped: this turn hit its time limit "
                    f"({self.settings.max_turn_seconds:.0f}s). "
                    + (
                        "Changed so far: " + ", ".join(self.ledger.changed_files)
                        if self.ledger.changed_files
                        else "Nothing was changed."
                    )
                    + " Ask again to continue.",
                    action_turn, turn_mutated, turn_ran_command,
                )
            if self.on_step is not None:
                self.on_step()
            response = self.llm.chat(
                [ChatMessage(role="system", content=self._system), *self.history],
                tools=self.registry.specs(),
                on_token=self.on_stream,
            )
            self.usage.add(response.usage)
            if self._cancelled():
                return self._finish_stopped()
            message = recover_inline_tool_call(response.message, self.registry.names())
            if (
                not message.tool_calls
                and message.content
                and self.settings.gate_constrained_retry
                and looks_like_tool_call(message.content, self.registry.names())
            ):
                retried = constrained_tool_retry(
                    self.llm,
                    [ChatMessage(role="system", content=self._system), *self.history],
                    self.registry.specs(),
                    self.registry.names(),
                )
                if retried is not None:
                    message, retry_usage = retried
                    self.usage.add(retry_usage)
                    self.recorder.event(
                        "chat", "constrained_tool_retry", tool=message.tool_calls[0].name
                    )

            if not message.tool_calls:
                content = normalize_markdown(
                    _SPECIAL_TOKEN_RE.sub("", message.content).strip()
                )
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
                # Detection always runs — even with a gate disabled — so an
                # ablation run still measures the violation it would have
                # caught. Only the CORRECTION is gated.
                violation, correction = None, None
                if action_turn:
                    if not turn_mutated and self._deflection(content):
                        violation = (
                            "pasted_code"
                            if _CODE_FENCE_RE.search(content)
                            else "promised_action"
                        )
                        correction = (
                            self._deflection(content)
                            if self.settings.gate_action
                            else None
                        )
                    elif not turn_ran_command and claims_verification(content):
                        violation = "false_verification"
                        correction = (
                            _FALSE_CLAIM_NUDGE
                            if self.settings.gate_false_verification
                            else None
                        )
                if violation:
                    self.recorder.event(
                        "chat",
                        "honesty_violation",
                        gate=violation,
                        corrected=bool(correction),
                    )
                if correction and corrections < _MAX_ACTION_NUDGES:
                    corrections += 1
                    self.recorder.event("chat", "action_gate_nudge", attempt=corrections)
                    self.history.append(message.model_copy(update={"content": content}))
                    self.history.append(ChatMessage(role="user", content=correction))
                    continue
                # A rename that missed a caller: checked ONCE here, not per
                # write. Renaming a definition necessarily breaks its callers
                # until the next edit repairs them, so blocking each write
                # traps the agent mid-operation.
                if turn_mutated and not dangling_checked:
                    dangling_checked = True
                    stale = self._dangling_references()
                    if stale:
                        self.recorder.event("chat", "dangling_reference", output="; ".join(stale))
                        self.history.append(message.model_copy(update={"content": content}))
                        self.history.append(
                            ChatMessage(
                                role="user",
                                content="The change is incomplete — these calls now "
                                "point at something that no longer exists:\n"
                                + "\n".join(f"- {s}" for s in stale)
                                + "\n\nFix every one of them now with edit_file.",
                            )
                        )
                        continue

                # Requirement coverage: the turn may not end while a part of
                # the request is provably absent from the diff. Checked
                # against evidence, never against the model's summary.
                if (
                    self.settings.gate_coverage
                    and action_turn
                    and turn_mutated
                    and self.effort != "fast"
                    and coverage_passes < _MAX_COVERAGE_PASSES
                    and looks_multi_requirement(user_text)
                ):
                    if requirements is None:
                        requirements = decompose(
                            self._thinker or self.llm, user_text, self.usage
                        )
                        if len(requirements) > 1:
                            self.recorder.event(
                                "chat", "requirements", count=len(requirements),
                                output="; ".join(r.text for r in requirements)[:500],
                            )
                    # a single-requirement request cannot be partially covered
                    if len(requirements) > 1:
                        evidence = build_evidence(
                            self.ledger.unified_diff(),
                            self.ledger.changed_files,
                            commands_run,
                        )
                        verdict = assess(
                            self._thinker or self.llm, requirements, evidence, self.usage
                        )
                        missing = verdict.unmet(requirements)
                        if missing:
                            coverage_passes += 1
                            self.recorder.event(
                                "chat", "coverage_gap",
                                unmet=[r.text for r in missing],
                                attempt=coverage_passes,
                            )
                            self.history.append(
                                message.model_copy(update={"content": content})
                            )
                            if self.settings.gate_focused_retry:
                                # Give each missing requirement its own clean
                                # turn instead of a nudge appended to a long,
                                # polluted history — the same model follows a
                                # short focused task it ignores at the end of
                                # a twenty-message conversation.
                                done = [r for r in requirements if r not in missing]
                                for requirement in missing:
                                    self._focused_pass(requirement, done)
                                self.history.append(
                                    ChatMessage(
                                        role="user",
                                        content="The missing parts were handled in "
                                        "focused passes. Summarize the whole change "
                                        "in plain text.",
                                    )
                                )
                            else:
                                self.history.append(
                                    ChatMessage(
                                        role="user", content=coverage_nudge(missing)
                                    )
                                )
                            continue

                if self.effort == "genius" and action_turn and not genius_checked:
                    # highest level: one completeness pass before accepting —
                    # the model must re-read the request and close any gaps
                    genius_checked = True
                    self.recorder.event("chat", "genius_check")
                    self.history.append(message.model_copy(update={"content": content}))
                    self.history.append(ChatMessage(role="user", content=_GENIUS_CHECK))
                    continue
                self.history.append(message.model_copy(update={"content": content}))
                return self._finish_turn(
                    content, action_turn, turn_mutated, turn_ran_command, append=False
                )

            self.history.append(message)
            for call in message.tool_calls:
                self.recorder.event("chat", "tool_call", tool=call.name, arguments=call.arguments)
                result = self.registry.execute(call.name, call.arguments)
                if result.ok and call.name in _FILE_MUTATING_TOOLS:
                    turn_mutated = True
                if result.ok and call.name in ("run_command", "run_powershell"):
                    turn_ran_command = True
                    command = str(call.arguments.get("command", ""))[:200]
                    if command:
                        commands_run.append(command)
                self.recorder.event(
                    "chat",
                    "tool_result",
                    tool=call.name,
                    ok=result.ok,
                    error=result.error,
                    output=(result.output[:400] if result.ok else None),
                )
                self.history.append(
                    ChatMessage(
                        role="tool",
                        content=result.render(self.settings.max_tool_output_chars),
                        tool_name=call.name,
                    )
                )

        return self._finish_turn(
            "Stopped: step budget exhausted for this turn.",
            action_turn, turn_mutated, turn_ran_command,
        )

    def set_effort(self, level: str) -> None:
        if level not in ("fast", "smart", "genius"):
            raise ValueError(f"Unknown effort level {level!r} (fast|smart|genius).")
        self.effort = level
        self.recorder.event("chat", "effort_changed", output=level)

    def _step_budget(self) -> int:
        base = self.settings.max_agent_steps
        if self.effort == "fast":
            return max(8, int(base * 0.6))
        if self.effort == "genius":
            return int(base * 1.5)
        return base

    @staticmethod
    def _issue_without_line(problem: str) -> str:
        _, _, rest = problem.partition(": ")
        return rest or problem

    def _dangling_references(self) -> list[str]:
        """Calls left pointing at definitions this session removed or renamed.

        Decided from the AST of each changed file against the copy the ledger
        saved before the first write — no model judgement, no guessing."""
        problems: list[str] = []
        for path, original in self.ledger.originals.items():
            if original is None or path.suffix != ".py" or not path.is_file():
                continue
            try:
                current = path.read_text(encoding="utf-8-sig", errors="replace")
            except OSError:
                continue
            name = path.name
            issues = dangling_reference_errors(original, current)
            # only NEW self-call breakage counts; a pre-existing one is not
            # this change's fault and must never trap the agent
            before = set(undefined_self_call_errors(original))
            issues += [
                issue
                for issue in undefined_self_call_errors(current)
                if self._issue_without_line(issue)
                not in {self._issue_without_line(b) for b in before}
            ]
            problems.extend(f"{name}: {issue}" for issue in issues)
        return problems[:4]

    def _focused_pass(
        self, requirement: Requirement, done: list[Requirement], budget: int = 6
    ) -> bool:
        """Run one requirement as a short task with a CLEAN context.

        Capability on fixed hardware comes from spending the model's
        attention better, not from a bigger model: the same 7B that ignores a
        correction at the end of a long conversation will carry out the same
        instruction when it is the only thing in front of it. Returns True if
        a file was changed."""
        self.recorder.event("chat", "focused_pass", output=requirement.text[:200])
        history: list[ChatMessage] = [
            ChatMessage(role="user", content=focused_prompt(requirement, done))
        ]
        changed = False
        for _ in range(budget):
            if self._cancelled():
                return changed
            if self.on_step is not None:
                self.on_step()
            try:
                response = self.llm.chat(
                    [ChatMessage(role="system", content=self._system), *history],
                    tools=self.registry.specs(),
                )
            except Exception:  # noqa: BLE001 — a focused pass must never kill the turn
                self.recorder.event("chat", "focused_pass_failed")
                return changed
            self.usage.add(response.usage)
            message = recover_inline_tool_call(response.message, self.registry.names())
            if not message.tool_calls:
                break
            history.append(message)
            for call in message.tool_calls:
                self.recorder.event(
                    "chat", "tool_call", tool=call.name, arguments=call.arguments
                )
                result = self.registry.execute(call.name, call.arguments)
                self.recorder.event(
                    "chat", "tool_result", tool=call.name, ok=result.ok,
                    error=result.error,
                    output=(result.output[:400] if result.ok else None),
                )
                if result.ok and call.name in _FILE_MUTATING_TOOLS:
                    changed = True
                history.append(
                    ChatMessage(
                        role="tool",
                        content=result.render(self.settings.max_tool_output_chars),
                        tool_name=call.name,
                    )
                )
        self.recorder.event("chat", "focused_pass_done", ok=changed)
        return changed

    def _interpret(self, user_text: str) -> str:
        """Two-model brain: the thinker model turns the raw message into a
        precise brief for the coder. An enhancement, never a blocker — any
        failure silently falls back to the raw message. Fast skips it;
        genius self-briefs with the main model when no thinker is set."""
        if self.effort == "fast":
            return ""
        thinker = self._thinker
        if thinker is None and self.effort == "genius":
            thinker = self.llm  # self-brief: reason first, act second
        if thinker is None:
            return ""
        try:
            response = thinker.chat(
                [
                    ChatMessage(role="system", content=_THINKER_SYSTEM),
                    ChatMessage(role="user", content=user_text),
                ]
            )
        except Exception:  # noqa: BLE001 — thinker failure must not kill the turn
            self.recorder.event("chat", "thinker_failed")
            return ""
        self.usage.add(response.usage)
        brief = response.message.content.strip()
        if brief:
            # the UI renders this as a collapsible reasoning card, so keep
            # enough of it to be readable rather than a clipped fragment
            self.recorder.event("chat", "intent_brief", output=brief[:2000])
        return brief

    @staticmethod
    def _deflection(content: str) -> str | None:
        """Classify a would-be final reply on an action turn where nothing was
        changed: pasted code or a promise of future action gets the matching
        corrective nudge; None means the reply is acceptable."""
        if _CODE_FENCE_RE.search(content):
            return _PASTED_CODE_NUDGE
        if _PROMISE_RE.search(content):
            return _PROMISE_NUDGE
        return None

    def _finish_turn(
        self,
        content: str,
        action_turn: bool,
        mutated: bool,
        ran_command: bool,
        append: bool = True,
    ) -> str:
        """End the turn: record what happened and persist. Every exit path
        goes through here so the metrics never miss a turn."""
        if append:
            self.history.append(ChatMessage(role="assistant", content=content))
        self.recorder.event(
            "chat",
            "turn_finished",
            action_turn=action_turn,
            mutated=mutated,
            ran_command=ran_command,
            unverified_claim=(not ran_command and claims_verification(content)),
            deflected=bool(action_turn and not mutated),
        )
        self.save_transcript()
        return content

    def _cancelled(self) -> bool:
        return self.should_stop is not None and self.should_stop()

    def _finish_stopped(self) -> str:
        note = "Stopped by user."
        self.history.append(ChatMessage(role="assistant", content=note))
        self.save_transcript()
        return note

    def _expand_mentions(self, text: str) -> str:
        """Inline @path/to/file.ext mentions. Image
        mentions are described by the vision model instead of inlined raw."""
        blocks: list[str] = []
        for mention in dict.fromkeys(_MENTION_RE.findall(text)):
            try:
                resolved = self._guard.resolve_path(mention.replace("\\", "/"))
            except Exception:  # noqa: BLE001 — bad mention is just text
                continue
            if not resolved.is_file():
                continue
            if resolved.suffix.lower() in IMAGE_EXTENSIONS:
                if self._vision is None:
                    blocks.append(
                        f"--- {mention} is an image; no vision model is configured "
                        "(set FORGE_VISION_MODEL, e.g. after `ollama pull llava`) ---"
                    )
                else:
                    seen = self._vision.run(path=mention)
                    blocks.append(
                        seen.output
                        if seen.ok
                        else f"--- could not see {mention}: {seen.error} ---"
                    )
                continue
            content = resolved.read_text(encoding="utf-8-sig", errors="replace")
            if len(content) > _MAX_MENTION_CHARS:
                content = content[:_MAX_MENTION_CHARS] + "\n... [truncated]"
            blocks.append(f"--- content of {mention} ---\n{content}")
        if blocks:
            text += "\n\n" + "\n\n".join(blocks)
        return text

    # -- compaction ----------------------------------------------------------------

    def _estimated_tokens(self) -> int:
        """Cheap upper-bound estimate (~4 chars/token) of what the next request
        will occupy: system prompt plus the whole history."""
        chars = len(self._system) + sum(
            len(m.content) + sum(len(str(tc.arguments)) for tc in m.tool_calls)
            for m in self.history
        )
        return chars // 4

    def _maybe_compact(self, force: bool = False) -> bool:
        # compact on EITHER trigger: too many messages, or the estimated token
        # footprint nearing the context window (long tool outputs can blow the
        # context in far fewer messages than the count threshold)
        over_messages = len(self.history) > self.settings.chat_compact_threshold
        over_tokens = self._estimated_tokens() > int(self.settings.num_ctx * 0.75)
        if not force and not over_messages and not over_tokens:
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
