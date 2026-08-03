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

## ✅ Milestone 11 — Grounded and unbound
Completes the grounding roadmap from Milestone 10 and removes the
Ollama-only limitation:
- **Syntax gate** (`tools/syntax_check.py`): every write/edit is parsed BEFORE
  it touches disk — Python (stdlib ast, compiler-grade messages), JSON, TOML,
  and tree-sitter languages (JS/TS/Go/Rust/Java/C/C++/C#/Ruby/PHP). A change
  that would introduce a syntax error is refused with the parser's diagnosis;
  only NEW errors are blocked, so already-broken files never trap the model.
  `FORGE_SYNTAX_GATE=0` disables.
- **Grammar-constrained tool-call retry**: when a reply looks like a mangled
  tool call, one re-ask constrained to the tool-call JSON schema (Ollama
  structured outputs / OpenAI json_schema) — malformed calls become
  impossible under the grammar. Wired into both the agent loop and chat.
- **Retrieval pre-flight**: the code most relevant to the task/turn is
  auto-injected before the model generates (BM25-gated: small talk injects
  nothing; @file mentions suppress it), so the model works from the
  repository's real APIs instead of remembered ones.
- **OpenAI-compatible provider** (`llm/openai_compat.py` + factory):
  `FORGE_PROVIDER=openai` points Forge at LM Studio, llama.cpp server, vLLM,
  OpenRouter, or any /v1/chat/completions endpoint — native tool calling,
  SSE streaming, structured outputs with graceful degrade, synthesized
  tool_call_ids. Ollama remains the default.
- **Token-aware compaction**: chat history compacts when the estimated token
  footprint nears `num_ctx`, not just on message count — long tool outputs
  can no longer blow the context window.

### Next grounding mechanisms (roadmap)
- **Verifier-guided best-of-N** — sample k candidate actions, keep the one that
  compiles / passes checks (test-time compute for local models).
- **FIM editing** — use qwen-coder's fill-in-the-middle training instead of
  error-prone old_string/new_string for insertions.

## ✅ Milestone 12 — Act, don't tell
Kills the classic small-model deflection (pasting code into chat instead of
changing files) with enforcement, not hope:
- **Action gate**: on a turn that asks for a change, a final reply containing
  a code fence — while no mutating tool succeeded — is bounced back with a
  corrective nudge (max 2 per turn). Same for "I will now edit X" promises.
- **False-verification gate**: a reply claiming tests/checks ran when no
  command executed this turn is bounced: run them for real or say you didn't.
- **PowerShell tool** (`run_powershell`, Windows only): full PowerShell with
  the same safety guard, timeout, and permission gating as run_command.
- **Environment awareness**: OS, Python version, and available shells are in
  the system prompt, so the model uses the right syntax for the machine.
- Template-tag leakage (`<tool_response>`, `<tool_call>`) stripped from
  replies. Live-verified against qwen2.5-coder:7b: file actually written,
  real `unittest discover` run, honest "no tests found" report.
- **Retry diversity** (first slice of verifier-guided test-time compute):
  after a review rejection, each coder retry samples hotter
  (base → +0.3/attempt, capped at 0.9) instead of deterministically
  repeating the same failure.
- **Plan-path grounding**: planner target_files that don't exist in the
  repository are annotated "(new file — does not exist yet)" so the coder
  creates deliberately instead of trusting a hallucinated path.

## ✅ Milestone 13 — GitHub intelligence + professional UI
- **github_repo / github_file tools** (`tools/github.py`): analyze any GitHub
  repository without cloning — metadata, README, full file tree, then read
  individual files (branch/tag/commit refs). Friendly rate-limit and 404
  messages; `FORGE_GITHUB_TOKEN` unlocks 5000 req/h + private repos. The
  prompt teaches the analyze → clone into workspace → edit-locally workflow.
  Live-verified against the real GitHub API.
- **UI redesign** (`server/static/app.html`): the desktop app moved from
  neon-ember hacker styling to a professional, Claude-grade design system —
  warm charcoal palette (#262624 / #1f1e1b), terracotta accent (#d97757),
  Charter serif brand ("✳ Forge"), flat surfaces with 1px borders (no
  gradients/glows/grain), circular send button, refined tool cards, code
  blocks, modals, and a calmer welcome screen. All JS behavior (streaming,
  approvals, sessions, diff viewer) unchanged; browser-verified.

## ✅ Milestone 13.5 — Own identity + web search
- **Original logo**: custom minimal anvil mark (inline SVG, terracotta) —
  Forge's own identity across the wordmark, welcome screen, message tags,
  and working indicator. No borrowed glyphs.
- **web_search tool**: DuckDuckGo HTML search (no API key) returning titles,
  decoded URLs, and snippets — pairs with fetch_url for read-the-docs flows.
  Numeric HTML entities now decoded. Live-verified.

## ✅ Milestone 14 — Two-model brain + live file cards
- **Dual-LLM pipeline** (`FORGE_THINKER_MODEL`): a reasoning model interprets
  each message (intent, steps, likely files, verification) and briefs the
  coder model, which implements with tools. Failures degrade silently to
  single-model mode; the brief is recorded as an `intent_brief` event.
- **File-operation cards** in the app: edits/creates/deletes render as
  "✎ Edited path ⌄" with a rotating chevron that expands into a real
  colored +/- diff (capped with "… N more lines"); success results stay
  quiet, errors surface inline. Verified by rendering live synthetic events.
- **New abstract logo**: a concave four-point spark (single SVG path, own
  design) replacing the anvil; it rotates/fades — "thinks" — while Forge is
  working and while a reply is streaming.

## ✅ Milestone 15 — Vision + effort levels
- **read_image tool** (`tools/vision.py`): a local multimodal model (llava,
  minicpm-v, …) describes images and transcribes their text for the coder —
  screenshots, mockups, diagrams, error photos. `FORGE_VISION_MODEL`;
  friendly "ollama pull" guidance when the model is missing.
- **@image mentions**: mentioning an image file in chat auto-runs the vision
  model and inlines the description before the coder generates.
- **Effort levels** — Fast / Smart / Genius (`FORGE_EFFORT`, UI switcher on
  welcome + top bar, `/api/chat/effort` to change mid-session):
  fast = no brief/pre-flight, 0.6× steps; smart = full grounding (default);
  genius = intent brief (self-brief without a thinker), one completeness
  check pass that re-reads the request before the final answer, 1.5× steps.

## Milestone 16 — Execution isolation & scale
- Docker sandbox for `run_command` (Docker SDK), per-run containers
- PostgreSQL replaces SQLite behind `MemoryStore`
- OpenTelemetry tracing; Prometheus metrics
- Multi-repo, multi-run concurrency

## Milestone 17 — More agents & evaluation
- Research, Documentation, Git (branch/PR), Testing agents
- Evaluation engine: task success rate, patch quality, hallucination rate,
  historical metrics; SWE-bench-style benchmark harness
