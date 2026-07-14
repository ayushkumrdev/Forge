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
- 69 tests, ~92% coverage; verified live against qwen2.5-coder:7b

## Milestone 2 — Repository intelligence
- Tree-sitter parsing for TypeScript/JavaScript/Go/Rust/Java (symbols, imports)
- Import & dependency graphs (networkx first; Neo4j behind the same interface)
- Symbol lookup tool for agents (`find_symbol`, `who_imports`)
- Persistent index cache in `.forge/` with invalidation on file change

## Milestone 3 — Retrieval
- Hybrid retrieval: BM25 (rank-bm25) + dense embeddings (Ollama embeddings,
  `nomic-embed-text`) over chunked code
- Qdrant as the vector store (embedded/local mode), AST-aware chunking
- `search_code` tool so the coder retrieves context instead of scanning blindly

## Milestone 4 — Memory layers
- Project memory: persisted repository understanding across runs
- Execution memory: surface past failures/fixes for similar tasks at plan time
- Knowledge memory: architecture decisions recorded and queryable

## Milestone 5 — Service + dashboard
- FastAPI backend: repository indexing, task submission, status, logs, metrics
- WebSocket streaming of run events
- Next.js + Tailwind + shadcn/ui dashboard: repo tree, execution timeline,
  live tool calls, diff viewer, run history, metrics
- Runs move from in-process to a worker (Celery/Redis) behind the same
  `ExecutionLoop` interface

## Milestone 6 — Execution isolation & scale
- Docker sandbox for `run_command` (Docker SDK), per-run containers
- PostgreSQL replaces SQLite behind `MemoryStore`
- OpenTelemetry tracing; Prometheus metrics
- Multi-repo, multi-run concurrency

## Milestone 7 — More agents & evaluation
- Research, Documentation, Git (branch/PR), Testing agents
- Evaluation engine: task success rate, patch quality, hallucination rate,
  historical metrics; SWE-bench-style benchmark harness
