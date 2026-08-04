# Forge

**Forge is a local autonomous AI software engineer** — an agent that runs
entirely on your machine, powered by [Ollama](https://ollama.com) and `qwen2.5-coder:7b`
by default, or any OpenAI-compatible server (LM Studio, llama.cpp, vLLM, OpenRouter, …)
via `FORGE_PROVIDER=openai`. Point it at a repository, give it a task in plain English,
and it plans the work, edits files through safe tools, runs your checks, reviews its own
diff, and iterates until the change passes review.

```
forge app                                # desktop app: chat window + folder picker
forge chat --repo path/to/project        # same experience in the terminal
forge run "add input validation to the signup endpoint" --repo path/to/project --check "pytest -q"
```

## The desktop app

`forge app` opens Forge as a native window (`forge shortcut` puts it on your
desktop). Pick a project folder to give Forge access, then just talk:

- Just talk — Forge reads, searches, edits and builds in the selected
  folder, showing every tool action live in the thread.
- **Everything in the chat box**: ＋ to attach images (described by the vision
  model), Ask first/Auto permission pill, Fast·Smart·Genius effort selector.
- **Conversations persist**: past chats appear in the sidebar across launches,
  resume with full history, and delete with one click.
- **Ask mode** (default): every file write or command pops an approval dialog —
  Deny / Allow / Allow always. Auto mode skips the prompts.
- **Switch models anytime** from the dropdown — it lists whatever `ollama list`
  has. Running a 7B today and a bigger model after a GPU upgrade is one click.
- Diff viewer and one-click **Undo all** for everything the session changed.
- Change folders anytime; each folder keeps its own memory and history under
  `.forge/`. Everything runs 100% locally.

## Chat mode

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
- **Coder** calls tools in a loop — `read_file`,
  `edit_file`, `write_file`, `append_to_file`, `delete_file`, `grep`,
  `find_files`, `list_dir`,
  `run_command`, `run_powershell` (Windows), `git`, plus code intelligence
  (`find_symbol`, `who_imports`),
  hybrid retrieval (`search_code`), web search (`web_search`), web docs
  (`fetch_url`), and GitHub (`github_repo`, `github_file`) — reading code
  before changing it and making minimal targeted edits.
- **Reviewer** is independent from the coder. It judges the *actual unified diff* and
  the results of your `--check` commands, then approves or returns concrete issues
  that are fed back to the coder for another attempt (up to `FORGE_MAX_REVIEW_CYCLES`).

## Grounding — hallucination attacked structurally

Small local models drift; Forge corrects them with reality, not prompting:

- **Syntax gate** — every write/edit is parsed *before* it touches disk (Python
  ast, JSON, TOML, tree-sitter for JS/TS/Go/Rust/Java and more). A change that
  would introduce a syntax error is refused with the parser's diagnosis; files
  that were already broken are never blocked. The repo can't silently rot.
- **Self-repairing edits** — near-miss `old_string`s are matched
  whitespace/CRLF-tolerantly against the file's real bytes, or answered with
  the closest actual snippet to copy. No retry death-spirals.
- **Grammar-constrained recovery** — a mangled tool call triggers one re-ask
  constrained to the tool-call JSON schema (structured outputs), so malformed
  calls are grammatically impossible.
- **Retrieval pre-flight** — the code most relevant to the task is auto-injected
  before the model generates, so it works from your real APIs, not remembered
  ones.
- **Act-don't-tell enforcement** — when you ask for a change, a reply that
  pastes code into chat (or promises "I will now edit…") while no file was
  touched is bounced back until the model actually does the work; claiming
  "tests passed" without having run a command is bounced the same way.

## Repository intelligence & retrieval

- Symbols for Python (stdlib `ast`) and TypeScript/JavaScript/Go/Rust/Java and more
  (tree-sitter, with regex fallback), cached per file in `.forge/repo_index.json`.
- An import graph (Python + JS/TS resolution) powers `who_imports` — blast-radius
  checks before an edit.
- Hybrid search: symbol-aware chunking, BM25 with a code-aware tokenizer, optional
  dense embeddings via Ollama (`FORGE_EMBEDDING_MODEL=nomic-embed-text`), fused
  with reciprocal-rank fusion.

## Vision — Forge can see

Pull a local multimodal model and Forge gains eyes:

```powershell
ollama pull llava:7b
$env:FORGE_VISION_MODEL = "llava:7b"
```

- Mention an image in chat — `what's wrong in @screenshot.png` — and the
  vision model describes it (text transcribed exactly) before the coder acts.
- Or the agent calls `read_image` itself: screenshots, UI mockups, diagrams,
  photos of error screens. Images never leave the machine.

## Effort levels — Fast · Smart · Genius

Pick how hard Forge thinks (welcome screen or top bar, switchable mid-chat):

| Level | What you get |
| --- | --- |
| **Fast** | Snappy: no intent brief, no retrieval pre-flight, smaller step budget |
| **Smart** | Default: retrieval pre-flight + grounding gates + thinker (if configured) |
| **Genius** | Highest: dual-model intent brief (self-briefs when no thinker is set), a final completeness check that re-reads your request before answering, 1.5× step budget |

## Two-model brain

Set `FORGE_THINKER_MODEL` to run Forge on two local models at once:

```powershell
$env:FORGE_THINKER_MODEL = "qwen2.5:7b-instruct"   # thinks
$env:FORGE_MODEL         = "qwen2.5-coder:7b"      # implements
```

Every message first goes to the **thinker**, which works out your intent —
what you actually want, the steps, the likely files, how to verify — and
writes a brief. The **coder** then implements with tools, guided by that
brief. Thinker failures degrade silently to single-model mode.

## GitHub intelligence

Point Forge at any GitHub project — no clone needed to understand it:

- `github_repo` pulls a repository's metadata, README, and full file tree in
  one call — an instant architectural picture.
- `github_file` reads any source file from it (branch/tag/commit selectable).
- To **modify** a GitHub project, Forge analyzes it, clones it into the
  workspace with git, then edits locally through the normal safe tool set.
- Works unauthenticated (60 req/h); set `FORGE_GITHUB_TOKEN` for 5000/h and
  private repositories.

## The verification ladder

Every change Forge writes climbs an ordered set of checks before it lands —
cheapest first, stopping at the first failure, and reporting *that* check's
own diagnostic so the model can actually fix it:

| rung | check | cost |
| --- | --- | --- |
| **L1 syntax** | the file parses (ast / JSON / TOML / tree-sitter) | ~10 ms |
| **L2 resolution** | every import and imported name really exists | ~50 ms |
| **L3 types** | `pyright`/`mypy` accepts it (`FORGE_GATE_TYPES=1`) | ~seconds |

L2 is the one that catches what nothing else does. This file is *valid
Python* — and refused:

```python
from utils import make_magic     # utils.py has no make_magic
```
```
Rejected — the resolution check failed: line 1: 'utils' does not define
'make_magic' (did you mean 'make_music'?). Nothing was written to main.py.
```

Only **new** failures block: a file that already had a broken import is
never blamed for it, so Forge can still work inside a half-finished refactor.

## Measuring the agent

Forge ships its own benchmark. `forge eval` runs **SWE-micro** — tiered tasks
on real fixture repositories, scored by hidden pytest suites the agent never
sees — and reports not just whether it succeeded but *how it behaved*:

```powershell
forge eval --tier 1 --seeds 3                  # baseline
forge eval --ablation no-gates                 # same tasks, gates disabled
```

| metric | meaning |
| --- | --- |
| **TSR** | hidden checks pass |
| **ADT** act-don't-tell | change requests where it actually changed something |
| **FVR** false-verification | claims tests ran when no command ran (lower better) |
| **GER** grounded-edit | edits that landed on real, existing text |
| **WCR** wasted-cycle | tool calls repeating an identical earlier call (lower better) |

Every gate has a kill-switch (`FORGE_GATE_ACTION=0`, `FORGE_SYNTAX_GATE=0`, …)
so any mechanism can be ablated and attributed. Detection keeps running while
a gate is off, so a disabled gate still *measures* what it would have caught.
See [docs/RESEARCH_ROADMAP.md](docs/RESEARCH_ROADMAP.md).

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
- Writes/edits that would introduce a syntax error are refused before touching
  disk (the syntax gate above), so a bad model step can't break a working file.
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
| `FORGE_PROVIDER` | `ollama` | `ollama` or `openai` (any OpenAI-compatible server) |
| `FORGE_OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `FORGE_OPENAI_BASE_URL` | `http://localhost:1234/v1` | OpenAI-compatible endpoint (LM Studio default) |
| `FORGE_OPENAI_API_KEY` | (empty) | Bearer token, if the server needs one |
| `FORGE_MODEL` | `qwen2.5-coder:7b` | Chat model (needs tool-calling support) |
| `FORGE_NUM_CTX` | `16384` | Context window tokens |
| `FORGE_MAX_AGENT_STEPS` | `25` | Coder tool-loop budget per attempt |
| `FORGE_MAX_REVIEW_CYCLES` | `3` | Code→review iterations per task |
| `FORGE_COMMAND_TIMEOUT_S` | `300` | Timeout for each shell command |
| `FORGE_SYNTAX_GATE` | `1` | Parse-verify every write/edit before it lands |
| `FORGE_GITHUB_TOKEN` | (empty) | GitHub token for higher API limits + private repos |
| `FORGE_THINKER_MODEL` | (empty) | Second model that interprets intent before the coder acts |
| `FORGE_VISION_MODEL` | (empty) | Multimodal model that lets Forge see images (e.g. `llava:7b`) |
| `FORGE_EFFORT` | `smart` | Default effort level: `fast`, `smart`, or `genius` |
| `FORGE_SEARCH_CANDIDATES` | `1` | Attempts per requirement; >1 enables candidate search |
| `FORGE_MAX_TURN_SECONDS` | `300` | Wall-clock ceiling for one turn (0 disables) |

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
