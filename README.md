# Forge

**Forge is a local autonomous AI software engineer** — a Claude Code-style agent that runs
entirely on your machine, powered by [Ollama](https://ollama.com) and `qwen2.5-coder:7b`.
Point it at a repository, give it a task in plain English, and it plans the work, edits
files through safe tools, runs your checks, reviews its own diff, and iterates until the
change passes review.

```
forge run "add input validation to the signup endpoint" --repo path/to/project --check "pytest -q"
```

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
  Python symbols) into a small ordered task list with complexity estimates.
- **Coder** works like Claude Code: it calls tools in a loop — `read_file`,
  `edit_file`, `write_file`, `grep`, `find_files`, `list_dir`, `run_command`, `git` —
  reading code before changing it and making minimal targeted edits.
- **Reviewer** is independent from the coder. It judges the *actual unified diff* and
  the results of your `--check` commands, then approves or returns concrete issues
  that are fed back to the coder for another attempt (up to `FORGE_MAX_REVIEW_CYCLES`).

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
                   filesystem / terminal / search / git tools
  repo/            repository scanner: tree, language stats, Python symbols
  agents/          Planner, Coder, Reviewer + shared tool-loop engine
  orchestrator/    the plan → code → check → review → iterate loop
  memory/          SQLite run/event store
  cli.py           forge doctor | index | plan | run | history
tests/             unit + mocked end-to-end tests
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
