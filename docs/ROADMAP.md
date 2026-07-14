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

## Milestone 7 — Execution isolation & scale
- Docker sandbox for `run_command` (Docker SDK), per-run containers
- PostgreSQL replaces SQLite behind `MemoryStore`
- OpenTelemetry tracing; Prometheus metrics
- Multi-repo, multi-run concurrency

## Milestone 7 — More agents & evaluation
- Research, Documentation, Git (branch/PR), Testing agents
- Evaluation engine: task success rate, patch quality, hallucination rate,
  historical metrics; SWE-bench-style benchmark harness
