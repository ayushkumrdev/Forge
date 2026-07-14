# Forge Architecture

Milestone 1 delivers a working vertical slice of the autonomous engineering loop.
Every module is designed to be replaced independently as later milestones land.

## Design principles

1. **Interfaces over implementations.** Agents depend on `LLMClient`, not Ollama.
   The orchestrator depends on `ToolRegistry`, not concrete tools. The memory store
   exposes a storage-agnostic API backed by SQLite today, PostgreSQL later.
2. **Safety is structural, not advisory.** Tools cannot reach the filesystem or the
   shell without passing through `SafetyGuard`; writes cannot happen without the
   `ChangeLedger` recording a backup. There is no code path around them.
3. **Everything observable.** Every LLM exchange, tool call and verdict is emitted
   through one `Recorder` to the console, a JSONL trace and SQLite.
4. **Small, verifiable increments.** The reviewer judges the actual unified diff and
   real check-command output — never the coder's claims.

## Module map

| Module | Responsibility | Replaceable by (roadmap) |
| --- | --- | --- |
| `forge.config` | Env-overridable settings (pydantic-settings) | — |
| `forge.llm` | `LLMClient` interface, Ollama impl, mock, JSON extraction | any provider (OpenAI-compatible, llama.cpp, …) |
| `forge.safety` | Command blocklist, path confinement, VCS protection | policy engine |
| `forge.tools` | `Tool` interface + registry, `ChangeLedger`, fs/terminal/search/git tools | Docker executor, patch tool, browser |
| `forge.repo` | Scanner: tree, language stats, Python `ast` symbols | tree-sitter multi-language, Neo4j graphs |
| `forge.agents` | `ToolLoopAgent` engine; Planner / Coder / Reviewer | more agents (research, docs, git) |
| `forge.orchestrator` | plan → code → check → review → iterate loop | Celery/Temporal task graph |
| `forge.memory` | SQLite run + event store | PostgreSQL, plus vector memory (Qdrant) |
| `forge.telemetry` | Recorder: console + JSONL + store | OpenTelemetry exporter |
| `forge.cli` | typer CLI | FastAPI service + Next.js dashboard |

## The execution loop

```
forge run "task" --repo R --check "pytest -q"
  │
  ├─ MemoryStore.start_run  ──────────────► .forge/forge.db
  ├─ RepoScanner.scan(R) ─► RepoSnapshot (tree, languages, symbols)
  ├─ Planner.plan(request, snapshot.summary())
  │     └─ structured JSON: {summary, tasks[{id, title, description, files, complexity}]}
  │
  ├─ for task in plan.tasks:                         ┌───────────────┐
  │     attempt = 1..max_review_cycles               │  ToolRegistry │
  │       ├─ Coder.execute(task, feedback?) ────────►│ read_file     │
  │       │    tool loop until plain-text answer     │ write_file    │
  │       │    or step budget exhausted              │ edit_file     │
  │       ├─ run check commands (pytest, lint, …)    │ list_dir      │
  │       ├─ ChangeLedger.unified_diff()             │ grep          │
  │       ├─ Reviewer.review(task, diff, checks)     │ find_files    │
  │       │    approved ─► next task                 │ run_command   │
  │       │    rejected ─► feedback ─► retry         │ git           │
  │       └─ cycles exhausted ─► task rejected       └───────┬───────┘
  │                                                          │
  │                                              SafetyGuard + ChangeLedger
  │                                              (blocklist, confinement,
  │                                               backups in .forge/backups/)
  └─ RunReport ─► .forge/runs/<run_id>.json, console summary, exit code
```

## Agent contracts

- **Planner** and **Reviewer** return validated JSON (`structured_call`): the raw
  output is parsed leniently (fences, embedded objects), validated with pydantic,
  and on failure the validation error is sent back to the model for a bounded
  number of retries.
- **Coder** is a `ToolLoopAgent`: system prompt + task context in, tool calls
  executed and fed back until the model answers in plain text. Its system prompt
  enforces read-before-write, minimal diffs, and self-verification via
  `run_command`.
- **Independence:** the reviewer shares no conversation state with the coder. It
  sees only the task, the coder's final summary, the real diff, and check output.

## Stopping criteria

A run ends when every task is approved (success), when a task exhausts
`max_review_cycles` (rejected → failed/partial), when the coder exhausts
`max_agent_steps` in an attempt (the reviewer then sees an unchanged diff and
rejects), or when the backend errors (failed, with the error in the report).

## State on disk (per target repository)

```
.forge/
  forge.db            SQLite: runs + events
  logs/<run>.jsonl    full event trace
  runs/<run>.json     RunReport (plan, verdicts, diff, usage, duration)
  backups/<run>/...   pre-modification copies of every touched file
```
