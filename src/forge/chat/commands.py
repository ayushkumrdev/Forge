"""Slash commands for the chat REPL, Claude Code style. Pure logic — the CLI
layer only prints what comes back, so every command is unit-testable."""

from __future__ import annotations

from dataclasses import dataclass

from forge.chat.session import ChatSession

HELP_TEXT = """\
/help              show this help
/diff              show all file changes made in this session
/undo              revert every file change made in this session
/run <task>        hand a task to the full autonomous loop (plan+code+review)
/init              generate a FORGE.md project instructions file
/model [name]      show or switch the Ollama model
/compact           compact older conversation history now
/clear             clear the conversation history
/exit              leave the chat
@path/to/file      mention a file to include its content in your message"""


@dataclass
class CommandResult:
    text: str = ""
    should_exit: bool = False


def is_command(line: str) -> bool:
    return line.strip().startswith("/")


def execute(session: ChatSession, line: str) -> CommandResult:
    parts = line.strip().split(maxsplit=1)
    name = parts[0].lstrip("/").lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if name in ("exit", "quit", "q"):
        return CommandResult(text="bye.", should_exit=True)
    if name == "help":
        return CommandResult(text=HELP_TEXT)
    if name == "clear":
        session.clear()
        return CommandResult(text="Conversation cleared.")
    if name == "diff":
        diff = session.ledger.unified_diff()
        return CommandResult(text=diff or "No file changes in this session yet.")
    if name == "undo":
        restored = session.undo()
        if not restored:
            return CommandResult(text="Nothing to undo.")
        return CommandResult(text="Restored: " + ", ".join(restored))
    if name == "compact":
        return CommandResult(
            text="History compacted." if session.compact() else "Nothing to compact."
        )
    if name == "model":
        if not arg:
            model = getattr(session.llm, "model", "unknown")
            return CommandResult(text=f"Current model: {model}")
        if hasattr(session.llm, "model"):
            session.llm.model = arg
            return CommandResult(text=f"Switched model to {arg}.")
        return CommandResult(text="This LLM client does not support switching models.")
    if name == "run":
        if not arg:
            return CommandResult(text="Usage: /run <task description>")
        return CommandResult(text=_run_autonomous(session, arg))
    if name == "init":
        return CommandResult(text=_generate_instructions(session))
    return CommandResult(text=f"Unknown command /{name}. Try /help.")


def _run_autonomous(session: ChatSession, request: str) -> str:
    """Bridge into the full plan → code → check → review loop from chat."""
    from forge.memory.store import MemoryStore
    from forge.orchestrator.loop import ExecutionLoop
    from forge.telemetry import Recorder

    store = MemoryStore(session.workspace)
    run_id = store.start_run(request)
    loop = ExecutionLoop(
        workspace=session.workspace,
        llm=session.llm,
        settings=session.settings,
        recorder=Recorder(run_id, session.workspace, store=store),
        store=store,
        run_id=run_id,
        check_commands=[],
    )
    report = loop.run(request)
    store.finish_run(run_id, report.status, report.model_dump())
    store.close()
    session.usage.add(report.usage)

    lines = [f"Autonomous run {run_id}: {report.status.upper()} ({report.duration_s}s)"]
    for result in report.task_results:
        lines.append(f"  [{result.status}] task {result.task_id}: {result.title}")
        if result.review and result.review.issues:
            lines.extend(f"    - {issue}" for issue in result.review.issues)
    if report.changed_files:
        lines.append("Changed: " + ", ".join(report.changed_files))
    if report.error:
        lines.append(f"Error: {report.error}")
    return "\n".join(lines)


def _generate_instructions(session: ChatSession) -> str:
    """Create FORGE.md (like Claude Code's /init creating CLAUDE.md)."""
    target = session.workspace / "FORGE.md"
    if target.exists():
        return "FORGE.md already exists — edit it directly or delete it first."
    from forge.llm.base import ChatMessage

    response = session.llm.chat(
        [
            ChatMessage(
                role="system",
                content="Write a concise FORGE.md project guide for an AI coding "
                "agent working in this repository: what the project is, key "
                "directories, how to run tests, and conventions to follow. "
                "Markdown, under 300 words. Output only the file content.",
            ),
            ChatMessage(role="user", content=session.snapshot.summary(8_000)),
        ]
    )
    session.usage.add(response.usage)
    target.write_text(response.message.content.strip() + "\n", encoding="utf-8")
    return (
        f"Wrote {target.name} ({len(response.message.content)} chars). "
        "It will be loaded next session."
    )
