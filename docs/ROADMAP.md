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

## ✅ Milestone 6 — Interactive chat
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
- Refined UI: forge-themed dark design, expandable tool cards with
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
  neon-ember hacker styling to a professional design system —
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

## ✅ Milestone 16 — Composer redesign + real conversations
- **Composer controls**, all inside the input box: ＋ image attach, Ask
  first/Auto pill, Fast·Smart·Genius selector, Stop and ↑ send — all inside
  the input box; top bar reduced to folder, model, Diff, Undo all; welcome
  screen simplified to just folder + model.
- **＋ attach images**: file picker → uploaded to `.forge/uploads/` in the
  workspace → auto-mentioned so the vision model describes them; removable
  chips shown above the input.
- **Ask/Auto switchable mid-chat** (`POST /api/chat/mode`) — the permission
  policy now carries its approver in both modes.
- **Persistent conversations**: transcripts in `.forge/chat/` are listed in
  the sidebar across app launches (empty ones skipped — no more phantom
  "New conversation" rows), resume rebuilds the visible history (injected
  context and corrective nudges stripped), and every row has a hover ✕
  delete that removes the session and its transcript.

## ✅ Milestone 17 — Evaluation harness (research Phase 1)
The instrument that turns "it works" into measured claims — see
[RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md) for the full programme.
- **SWE-micro benchmark** (`evals/suite.py`): 7 tiered tasks (T1 single-file,
  T2 cross-file, T3 repo-level) on fixture repos, scored by hidden pytest
  suites written only after the agent finishes and deleted afterwards.
- **Behavioural metrics** (`evals/metrics.py`) derived from event traces:
  ADT, FVR, GER, WCR, tool reliability. Unobserved rates report `None`, never
  a misleading 0.
- **Ablation infrastructure**: every gate has a kill-switch; six presets from
  `all-gates` to `no-gates`. Detectors keep running when gates are off, so
  ablations still measure the violations they stop preventing.
- **`forge eval`** CLI with tiers, seeds, ablations and JSON reports.
- Fixes the harness found on its first live run: a false-verification metric
  that scored honest disclaimers as lies, an action detector blind to
  inflected verbs, and an **edit death-spiral caused by over-escaped tool
  arguments** — now repaired (`unescape_literals`), cutting that task from
  16 tool calls to 2 and the suite from 85s to 48s.

## ✅ Milestone 18 — Verification ladder (research contribution C1)
- **`forge.verify.ladder`**: every mutation climbs L1 syntax → L2 resolution
  → L3 types, cheapest first, stopping at the first failure and returning
  that rung's own diagnostic (not a generic refusal) so the model recovers
  instead of retrying blindly.
- **L2 resolution is the novel rung**: a file can parse perfectly and still
  import something that doesn't exist — the most characteristic code-model
  hallucination. Resolves imports against stdlib / installed packages / the
  repository, and checks imported *names* exist in repo modules, with a
  "did you mean" for near-misses.
- Invariants (tested): only NEW failures block, line drift never resurrects
  a pre-existing problem, unparseable targets are never guessed at.
- Wired into `write_file` / `edit_file`; `FORGE_GATE_RESOLUTION=0` and the
  `no-resolution` ablation isolate it. `FORGE_GATE_TYPES=1` enables L3.
- **`forge sweep`** runs the benchmark under every gate configuration and
  prints the ablation table.

## ✅ Milestone 19 — Requirement coverage + the tool the benchmark demanded
- **T2 benchmark tier**: multi-requirement work in a SINGLE file, filling the
  gap between one edit (67% solved) and cross-file (0/6). Tiers now run
  T1 single edit / T2 several requirements / T3 cross-file / T4 repo-level.
- **`verify/coverage.py` (C9)**: splits a request into atomic requirements and
  judges each against the REAL diff and the commands that ran — never the
  model's summary. The turn cannot end while a requirement is provably absent.
  Cheap regex pre-filter so single-outcome requests never pay for it;
  "cannot tell" is never treated as "unmet"; bounded at 2 passes.
- **`append_to_file`**: the missing affordance. The benchmark showed the model
  calling edit_file with an empty old_string (meaning "append"), being
  refused, and giving up with nothing written. Adding it took tier 2 to
  ADT 100%, tool reliability 100%, grounded-edit 100%, FVR 0%.
- **ADT metric fixed**: running a command counted as "acting", so the metric
  was inflated and the act-don't-tell gate would not fire on a model that
  merely re-ran the tests. It now counts only file-changing tools.

## ✅ Milestone 20 — Bug hunt + execution-guided candidate search
Overall TSR 23.8% -> 35.0%; tier 1 56% -> 83%; tier 2 0/6 -> 2/6, and 44.4%
with search on (matched control 22.2%).
- **Line endings**: every file Forge wrote on Windows was converted LF->CRLF,
  so a one-line edit produced a whole-file diff. Writes now preserve the
  file's own style. The single most damaging bug found, and it never showed
  up in a metric.
- **Syntax gate strengthened**: used `ast.parse`, which ACCEPTS return/break/
  yield outside their block; now `compile()`.
- **Indentation repair**: multi-line edit replacements arrived at column zero
  and dedented code out of its function; now shifted to the anchor's depth,
  kept only if the result verifies.
- **`append_to_file`**: the missing affordance — the model reached for
  edit_file with an empty old_string to append, was refused, and gave up.
- **Undefined self-calls + missed callers**: a botched rename leaves either a
  call to nothing or a caller pointing at the old name. Both decided from the
  AST. Checked ONCE at turn end — gating per-write trapped the agent and cost
  3/6 -> 1/6.
- **Turn time limit** (`FORGE_MAX_TURN_SECONDS`): a turn was bounded in steps
  but not time; one task ran 611 seconds.
- **`verify/search.py`**: k attempts per requirement at diverse temperatures,
  each isolated by a byte-exact snapshot, scored by requirement coverage and
  the ladder, best kept. `FORGE_SEARCH_CANDIDATES`.
- Decomposition fixed: it had been emitting overlapping and unfalsifiable
  requirements, which made search *worse* than no search until repaired.

## ✅ Milestone 21 — rename_symbol, candidate search, and a broken benchmark
Overall TSR 23.8% -> **45.0%**; tier 2 0/6 -> **6/9**.
- **`rename_symbol`**: renaming as ONE correct AST operation. Text
  substitution is the wrong instrument — asked to rename `pop`, the model
  issued edit_file with "pop()", which matched `self.pop()` (the CALL)
  instead of `def pop(self)`, and never recovered. Comments, strings and
  identifiers that merely share the name are untouched; attribute renames
  follow only a plain receiver, so a list's own `.pop()` is left alone.
- **Candidate search** (`verify/search.py`, `FORGE_SEARCH_CANDIDATES`):
  k attempts per requirement at diverse temperatures, isolated by byte-exact
  snapshots, scored by requirement coverage and the ladder, best kept.
  Matched control on tier 2: k=1 2/9, k=3 4/9.
- **The benchmark itself was broken**: `t2-rename-in-file` imported the
  stdlib `queue` instead of the fixture's `taskqueue`, so it could never pass
  no matter what Forge produced. Every task now carries a reference solution
  and a test proves all ten are solvable — a broken check is a failing test,
  not a permanent silent zero.
- **Method signatures**: a rename that drops `self` (`def enqueue(item)`)
  parses, resolves, and fails at every call. Now caught from the AST.
- Structural checks re-run after each fix attempt (bounded), and their nudge
  carries the file's current contents plus, for a rename, the exact
  rename_symbol call to make.

## ✅ Milestone 22 — Cross-file rename, coverage arming, tool diagnostics
Overall TSR **45.0% -> 55.0%**; every tier now scores (T1 5/6, T2 3/6,
T3 2/4, T4 1/4) — from 23.8% and three empty tiers at the start of the hunt.
- **`rename_symbol` propagates across the repository**: it finds every file
  importing the module and updates both shapes (`from m import f` plus the
  bare calls it binds, and `m.f(...)`). Each dependent file is verified
  before writing and ledgered for undo. Took t3-rename-propagate 0/2 -> 2/2.
- **Coverage gate arms without a conjunction**: a request with three
  requirements and no "and" was treated as single-outcome, so the gate never
  ran and the model shipped two of three. The pre-filter now counts clauses
  containing an action verb.
- **Tool argument errors ground the model**: missing/unexpected arguments are
  named in the tool's vocabulary instead of leaking a Python signature.

## ✅ Milestone 23 — Rigorous flaw pass (TSR 53.3% over 30 runs)
Measured on 3 seeds: T1 7/9, T2 4/9, T3 3/6, T4 2/6 — every tier scores,
from 23.8% and three empty tiers. ADT 90-100%, tool reliability 90-93%,
HIR 0%.
- **`run_tests`**: detects the project's runner instead of making the model
  guess. It ran `unittest discover` on a pytest-style suite, got "0 tests",
  and started writing its own. T4 0/6 -> 2/6.
- **`mechanically_unmet()`**: the coverage judge declared a requirement met
  while the file it named was absent from the diff. Evidence from the ledger
  now overrides the model's opinion.
- **rename_symbol collision check**: a dangling reference to the new name is
  a half-finished rename, not a collision — it was refusing its own repair.
- **Give-up gate**: an action turn ending with no change after a failed write
  is bounced. Permission denials are excluded, so Forge never argues with a
  user who said no.
- Tool argument errors name the missing parameter instead of leaking a Python
  signature; empty old_string on a missing file points at write_file.

## ✅ Milestone 24 — Plan-first execution + diff-aware structural checks
`t2-rename-in-file` scored for the first time (0/3 -> 1/3). Nothing about the
model changed; what changed is *when* its attention is spent.
- **Plan-first (`gate_plan_first`)**: a multi-part request is decomposed up
  front and each requirement runs as its own focused step with a clean
  context — before the model has a chance to half-finish part one and then
  reason from the mess. The focused pass already worked; it had only ever
  run as repair, after the damage.
- **Requirement-shaped tool guidance**: the focused prompt used to end with a
  fixed "use `append_to_file` or `edit_file`", which beat the system prompt's
  "always use `rename_symbol`" on every rename — and hand-editing a rename
  hits `self._items.pop(0)` while missing `self.pop()`. Guidance now follows
  the requirement, including warnings about boolean returns and about
  changing a signature nobody asked to change.
- **`narrowed_signature_errors`**: a public function that no longer accepts
  the calls it used to (`register(name, email)` shipped as `register(user)`).
  Diff-aware — the file parses and resolves; it is wrong only relative to
  what was there before. Replayed over 20 commits of real history: no false
  positives.
- **`inconsistent_boolean_return_errors`**: `return local and domain` gives
  `''`, not `False`. Only fires when the function also returns a literal
  bool *and* an operand is not already boolean.
- **Constrained tool retry in the focused pass**: the main loop has had it
  for a long time; the focused pass never did, and a mangled call there is
  worse because the pass *is* the step.
- The app shows the plan and marks each step, so multi-part work no longer
  arrives as a run of unexplained edits.

## ✅ Milestone 25 — Greenfield, L4, and the paper
Forge builds a project from an empty folder for the first time
(`t5-build-package`, three tool calls, 35 seconds).
- **T5 greenfield tier**: the agent starts in an empty directory. Baseline was
  0/2 with half of all tool calls failing, because every instruction Forge has
  assumes a repository to change. The prompt now says what an empty folder
  means, and `File not found` names what the folder really contains instead of
  leaving the model to guess the same wrong path again.
- **L4 runtime rung** (`verify/runtime.py`): imports every module the turn
  wrote, in a subprocess. Catches what no static rung can, such as a package
  whose `__init__.py` exports nothing. Found a real bug in Forge's own suite
  on the way in.
- **`unexported_package_errors`**: a newly created package that exports
  nothing its modules define. Only judges packages created in this session,
  since a namespace package is a legitimate style.
- **Plan-first is now limited, not extended**: an empty workspace skips
  decomposition. Splitting one package into six isolated steps made them fight
  each other. Decomposition helps for independent requirements and harms one
  coherent artifact.
- **L2 relaxed for a package under construction**: `from .primes import x`
  written before `primes.py` exists is correct, and refusing it pushed the
  model into an export-free `__init__.py`. Forge was causing the defect its
  own check reported.
- **Enforced verification**: when the model claims the checks ran and no
  command was issued, Forge runs them and hands back the real output.
- **Self-briefing measured and reverted**: it cost `t3-wire-validator` 2/3,
  took false verification from 14.3% to 47.6% and dropped act-don't-tell from
  100% to 94.3%. Off by default, with the measurement beside it.
- **`forge swebench`**: real SWE-bench Lite instances, patches generated
  locally and scored inside the official Docker images on the official
  criterion.
- **`docs/PAPER.md`**: the write-up, with every number traced to a run.

## Milestone 26 — Execution isolation & scale
- Docker sandbox for `run_command` (Docker SDK), per-run containers
- PostgreSQL replaces SQLite behind `MemoryStore`
- OpenTelemetry tracing; Prometheus metrics
- Multi-repo, multi-run concurrency

## Milestone 27 — More agents & evaluation
- Research, Documentation, Git (branch/PR), Testing agents
- Evaluation engine: task success rate, patch quality, hallucination rate,
  historical metrics; SWE-bench-style benchmark harness
