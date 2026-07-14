# Forge

**Forge is a local autonomous AI software engineer** — a Claude Code-style agent that runs
entirely on your machine, powered by [Ollama](https://ollama.com) and `qwen2.5-coder:7b`.
Point it at a repository, give it a task in plain English, and it plans the work, edits
files through safe tools, runs your checks, reviews its own diff, and iterates until the
change passes review.

```
forge chat --repo path/to/project        # interactive, Claude Code-style
forge run "add input validation to the signup endpoint" --repo path/to/project --check "pytest -q"
```

## Chat mode — the Claude Code experience

`forge chat` opens an interactive session on your repository:

- Converse naturally; Forge answers questions in text and makes changes with tools.
- **Permissions**: by default Forge asks before every file write or shell command
  (`--auto` to skip prompts). Read-only tools never prompt.
- **@file mentions** inline a file's content into your message: `explain @src/app.py`.
- **Slash commands**: `/diff` (session changes), `/undo` (revert them all),
  `/run <task>` (hand off to the full plan→code→review loop), `/init` (generate
  FORGE.md), `/model [name]`, `/compact`, `/clear`, `/help`, `/exit`.
- **FORGE.md** (or CLAUDE.md) at the repo root is loaded into the system prompt —
  project conventions the agent must follow.
- **Continuity**: history persists across turns, long conversations are compacted
  with an LLM summary, and `--resume` restores your previous session.

## How it works

```
             ┌─────────────────────────────────────────────────┐
             │                 Execution loop                  │
 request ──► │ scan repo ─► Planner ─► Coder ─► checks ─►      │ ──► report
             │                ▲        (tools)  Reviewer       │     + diff
             │                └── feedback on rejection ◄──────┘     + backups
             └─────────────────────────────────────────────────┘
```

- **Planner** turns your request plus a repository snapshot (file tree, languages,
  multi-language symbols) into a small ordered task list — informed by **execution
  memory**: lessons from similar past runs are recalled and injected so known
  mistakes aren't repeated.
- **Coder** works like Claude Code: it calls tools in a loop — `read_file`,
  `edit_file`, `write_file`, `grep`, `find_files`, `list_dir`, `run_command`, `git`,
  plus code intelligence (`find_symbol`, `who_imports`) and hybrid retrieval
  (`search_code`) — reading code before changing it and making minimal targeted edits.
- **Reviewer** is independent from the coder. It judges the *actual unified diff* and
  the results of your `--check` commands, then approves or returns concrete issues
  that are fed back to the coder for another attempt (up to `FORGE_MAX_REVIEW_CYCLES`).

## Repository intelligence & retrieval

- Symbols for Python (stdlib `ast`) and TypeScript/JavaScript/Go/Rust/Java and more
  (tree-sitter, with regex fallback), cached per file in `.forge/repo_index.json`.
- An import graph (Python + JS/TS resolution) powers `who_imports` — blast-radius
  checks before an edit.
- Hybrid search: symbol-aware chunking, BM25 with a code-aware tokenizer, optional
  dense embeddings via Ollama (`FORGE_EMBEDDING_MODEL=nomic-embed-text`), fused
  with reciprocal-rank fusion.

## Dashboard

```powershell
forge serve --repo path\to\project    # http://127.0.0.1:8321
```

A self-contained web dashboard (no CDN, works offline): submit runs, watch the live
agent event feed, inspect diffs, browse the repository snapshot and learned lessons.
The same REST API (`/api/runs`, `/api/repo`, `/api/memory`, `/api/health`, docs at
`/docs`) is usable programmatically.

## Safety, always on

- Destructive commands (`rm -rf /`, `git push --force`, `format`, `dd`, …) are blocked
  by a safety guard that every tool call passes through.
- File access is confined to the target repository; `.git/` internals are write-protected.
- **Every file Forge modifies is backed up first** to `.forge/backups/<run-id>/`.
- Edits require an exact unique text match, so files are never overwritten blindly.
- Every action (tool call, LLM exchange, review verdict) is logged to
  `.forge/logs/<run-id>.jsonl` and an SQLite store at `.forge/forge.db`.

## Quickstart

Requirements: Python 3.12+, [Ollama](https://ollama.com) running locally.

```powershell
ollama pull qwen2.5-coder:7b

git clone <this repo> && cd forge
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"     # Linux/macOS: .venv/bin/pip

forge doctor                               # verify Ollama + model
forge index  --repo path\to\project        # inspect what Forge sees
forge plan  "describe your change" --repo path\to\project
forge run   "describe your change" --repo path\to\project --check "pytest -q"
forge history --repo path\to\project       # past runs
```

## Configuration

Everything is overridable via environment variables (or a `.env` file):

| Variable | Default | Meaning |
| --- | --- | --- |
| `FORGE_OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `FORGE_MODEL` | `qwen2.5-coder:7b` | Chat model (needs tool-calling support) |
| `FORGE_NUM_CTX` | `16384` | Context window tokens |
| `FORGE_MAX_AGENT_STEPS` | `25` | Coder tool-loop budget per attempt |
| `FORGE_MAX_REVIEW_CYCLES` | `3` | Code→review iterations per task |
| `FORGE_COMMAND_TIMEOUT_S` | `300` | Timeout for each shell command |

## Project layout

```
src/forge/
  config.py        settings (env-overridable)
  telemetry.py     event recording: console + JSONL + SQLite
  llm/             provider-agnostic LLM interface; Ollama + mock clients
  safety/          command blocklist, path confinement, VCS protection
  tools/           tool interface, registry, change ledger (backups/diffs),
                   filesystem / terminal / search / git / code-intel /
                   retrieval tools
  repo/            scanner, multi-language symbols (tree-sitter), import graph
  retrieval/       BM25 + optional embeddings, symbol-aware chunking, RRF
  agents/          Planner, Coder, Reviewer + shared tool-loop engine
  orchestrator/    the plan → code → check → review → iterate loop
  memory/          SQLite run/event store + execution memory (lessons)
  chat/            interactive session: history, mentions, compaction,
                   slash commands, FORGE.md instructions
  server/          FastAPI API + bundled web dashboard
  cli.py           forge chat | run | serve | plan | index | memory | history | doctor
tests/             unit + mocked end-to-end tests (101 tests)
docs/              architecture and roadmap
```

Every module is replaceable behind its interface — see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design and
[docs/ROADMAP.md](docs/ROADMAP.md) for what lands in the next milestones
(tree-sitter indexing, hybrid retrieval, FastAPI service + dashboard, richer memory).

## Development

```powershell
.venv\Scripts\python -m pytest --cov=src/forge   # run tests
.venv\Scripts\python -m ruff check src tests     # lint
```

## License

MIT
