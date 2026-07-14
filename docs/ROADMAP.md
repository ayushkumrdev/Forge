# Forge Roadmap

Milestones are strictly incremental: each one ships working, tested code and
never requires rewriting a previous layer — only swapping an implementation
behind an existing interface.

## ✅ Milestone 1 — Working core (this release)
- Provider-agnostic LLM client (Ollama + qwen2.5-coder, native tool calling)
- Safety guard: command blocklist, path confinement, `.git` write protection
- Tool layer: filesystem (read/write/edit/list), terminal, grep/glob, git —
  with automatic backups and unified-diff change ledger
- Repository scanner: file tree, language stats, Python symbol index (stdlib ast)
- Planner / Coder / Reviewer agents + plan→code→check→review→iterate orchestrator
- SQLite execution memory, JSONL traces, rich console telemetry
- CLI: `doctor`, `index`, `plan`, `run`, `history`
- Inline tool-call recovery for local model templates that emit JSON-in-content
- 101 tests, ~93% coverage; full autonomous run (plan → code → test → review →
  approve, multi-file with imports) verified live against qwen2.5-coder:7b

## ✅ Milestone 2 — Repository intelligence
- Tree-sitter parsing for TypeScript/JavaScript/Go/Rust/Java + more (with
  regex fallback when tree-sitter is unavailable)
- Import graph with Python (absolute/relative/src-layout) and JS/TS resolution
- Symbol lookup tools for agents: `find_symbol`, `who_imports`
- Persistent index cache in `.forge/repo_index.json` keyed by mtime+size

## ✅ Milestone 3 — Retrieval
- Hybrid retrieval: pure-Python BM25 (code-aware tokenizer: camelCase/snake
  splitting, light stemming, stopwords) + optional Ollama dense embeddings
  (`FORGE_EMBEDDING_MODEL`), fused with reciprocal-rank fusion
- Symbol-aware chunking with windowed fallback
- `search_code` agent tool

## ✅ Milestone 4 — Execution memory
- Task outcomes persisted as lessons (request, task, verdict, reviewer issues)
- Similar past lessons recalled via BM25 into the planner prompt
- `forge memory` CLI command

## ✅ Milestone 5 — Service + dashboard
- FastAPI backend: health, repo snapshot, run submission with background
  execution, run reports, event traces, lessons (`/docs` for OpenAPI)
- Self-contained dark web dashboard: live run feed, diff viewer, repo and
  memory views; `forge serve`

## ✅ Milestone 6 — Interactive chat (the Claude Code experience)
- `forge chat`: persistent multi-turn tool-loop session on the repository
- Permission system: ask-before-write/command by default, `--auto` to skip;
  read-only tools never prompt; per-session "always allow"
- FORGE.md / CLAUDE.md project instructions loaded into system prompts
- Slash commands: /diff /undo /run /init /model /compact /clear /help /exit
- @file mentions, LLM history compaction, transcript resume, ledger-based undo

## ✅ Milestone 7 — Desktop app
- `forge app`: native desktop window (pywebview) with a full chat UI
- Folder picker (native dialog) scopes Forge's access; switch folders anytime
- Model switcher dropdown listing installed Ollama models — swap mid-session
- Ask/Auto permission modes with approval dialogs (Deny / Allow / Allow always)
- Live tool-activity feed, diff viewer, one-click undo of session changes
- `forge shortcut` installs a desktop shortcut

## ✅ Milestone 8 — App level-up
- Token streaming: replies render live as the model generates (NDJSON stream)
- Stop button (soft cancel between steps) and message queueing while busy
- Multiple conversations with a sidebar, titles, and instant switching
- Recent-folders memory on the welcome screen; session re-attach on reload
- Claude-grade UI: forge-themed dark design, expandable tool cards with
  arguments/results, code blocks with copy + syntax highlighting, live
  elapsed-time working indicator, token usage footer

## ✅ Milestone 9 — God prompt + more tools
- Rewrote the chat system prompt for small local models: explicit tool
  protocol with a worked example, prime directives (act don't instruct, verify
  everything, finish the whole request), a step-by-step workflow, and a
  recovery playbook for the failure modes 7B models hit
- New `delete_file` tool — ledgered and reversible with /undo, permission-gated
- New `fetch_url` tool — read web docs/API references (HTML stripped, read-only)
- Fixed the markdown/copy-button bug: multi-code-block replies now render with
  working ⧉ COPY buttons (with clipboard fallback + copied feedback)

## ✅ Milestone 10 — Grounding layer (verifier-in-the-loop)
Attacks hallucination structurally, not with prompting. Reality is the verifier:
- **Self-repairing edits** (`edit_repair.py`): three-tier match — exact, then
  whitespace/CRLF/indent-tolerant auto-apply to the file's REAL bytes, then a
  fuzzy "closest actual snippet" correction. Kills the `old_string not found`
  retry death-spiral that small models fall into (proven on the exact failure
  modes seen in earlier live runs).
- Never invents an edit location; only ever applies a replacement to text that
  provably exists in the file.

### Next grounding mechanisms (roadmap)
- **Grammar-constrained tool calls** via Ollama structured outputs — the model
  cannot emit malformed calls.
- **Retrieval pre-flight** — auto-inject the real relevant code before the model
  generates, so it can't hallucinate APIs.
- **Verifier-guided best-of-N** — sample k candidate actions, keep the one that
  compiles / passes checks (test-time compute for local models).
- **FIM editing** — use qwen-coder's fill-in-the-middle training instead of
  error-prone old_string/new_string for insertions.

## Milestone 11 — Execution isolation & scale
- Docker sandbox for `run_command` (Docker SDK), per-run containers
- PostgreSQL replaces SQLite behind `MemoryStore`
- OpenTelemetry tracing; Prometheus metrics
- Multi-repo, multi-run concurrency

## Milestone 7 — More agents & evaluation
- Research, Documentation, Git (branch/PR), Testing agents
- Evaluation engine: task success rate, patch quality, hallucination rate,
  historical metrics; SWE-bench-style benchmark harness
