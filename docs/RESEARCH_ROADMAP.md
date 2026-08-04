# Forge Research Roadmap — Verifier-Gated Autonomy

**The mega roadmap: from a working local agent to a publishable research
system.** This document is self-contained: thesis, novelty positioning,
ranked contributions with implementation sketches, the evaluation design a
reviewer will demand, and a paper skeleton. Work through it top to bottom.

---

## 1. The thesis (what makes Forge publishable)

Every serious coding agent today (Devin, OpenHands, SWE-agent, Aider,
Claude Code) assumes a **frontier-scale model**. Forge's bet is the opposite:

> **Reliability in autonomous software engineering can come from
> *structural verification* instead of *model scale*. A 7B local model
> wrapped in a stack of reality-checking gates can be made honest,
> grounded, and useful — on consumer hardware, with zero data leaving
> the machine.**

Forge already implements the seed of this thesis, live-verified against
qwen2.5-coder:7b:

| Mechanism (shipped) | Failure mode it kills | Where |
| --- | --- | --- |
| Syntax gate (parse-verify every write, auto-revert) | committing broken code | `tools/syntax_check.py` |
| Self-repairing edits (3-tier match + grounded correction) | `old_string` hallucination death-spiral | `tools/edit_repair.py` |
| Grammar-constrained tool-call retry | malformed tool JSON | `agents/structured.py` |
| Retrieval pre-flight | inventing APIs from memory | `retrieval/engine.py` |
| Action gate + promise detector | pasting code instead of acting | `chat/session.py` |
| False-verification gate | claiming "tests passed" without running them | `chat/session.py` |
| Plan-path grounding | planner hallucinating file paths | `orchestrator/loop.py` |
| Reviewer on real diffs + real check output | trusting the coder's claims | `agents/reviewer.py` |
| Two-model brain (thinker briefs coder) | intent misreading | `chat/session.py` |
| Effort levels (fast/smart/genius + completeness check) | premature "done" | `chat/session.py` |
| Escape repair (decode over-escaped tool arguments) | JSON escaping death-spiral | `tools/edit_repair.py` |

The research program below turns this from "engineering that works" into
"science with measured claims." **The single most important insight: nothing
below is publishable without Section 3 (the evaluation harness). Build it
first.**

---

## 2. Novelty positioning — know the prior art before claiming anything

A paper lives or dies on its delta. This is the honest map:

| Prior work | What it does | Forge's delta |
| --- | --- | --- |
| SWE-agent (Yang et al. 2024) | agent-computer interface for frontier models | Forge targets **sub-10B local** models; interface alone is insufficient there — gates are the contribution |
| OpenHands / AutoCodeRover | agent scaffolds, frontier-model assumption | same delta: scale-free reliability; plus measurable honesty gates |
| Aider | repo-map + edit formats, works with local models | Aider trusts the model's output; Forge **verifies before disk** and bounces deflection/false claims structurally |
| AlphaCodium / CodeT (2022-24) | test-based candidate reranking for *competitive programming* | Forge applies execution-guided selection to *repository editing trajectories*, not single-function synthesis |
| Voyager (2023) | skill library, but in Minecraft with no correctness pressure | Forge's skills are **admission-gated by real verification** on real repos |
| Speculative decoding (Leviathan 2023) | token-level draft/verify | Forge proposes **action-level** draft/verify (C3) — a different granularity, unexplored |
| Reflexion / self-refine | model critiques its own text | Forge's critic is **reality** (parsers, tests, diffs), not the model — that distinction is the paper |
| Agentless (2024) | fixed pipeline beats agents on SWE-bench | supports Forge's thesis (structure > free agency); Forge generalizes it to interactive agents |

**The claim to defend:** *no published system combines (a) sub-10B local
models, (b) a formalized cheap-to-expensive verification ladder, (c)
behavioral honesty gates with measured deflection/false-claim rates, and
(d) execution-guided trajectory selection — evaluated together with
ablations.* That combination + the metrics in §3 is the paper.

---

## 3. FIRST: the evaluation harness (Milestone 17 — do this before anything)

Without numbers there is no paper. Build `src/forge/evals/`:

**3.1 Benchmark: SWE-micro.** SWE-bench is calibrated for frontier models
(7Bs score ~0-3%, producing no signal). Create a graduated benchmark that
*discriminates* between small-model configurations:

- 60-100 tasks across 3 tiers: T1 single-file edits (add function, fix off-by-one),
  T2 cross-file (rename + update importers, add feature touching 2-3 files),
  T3 repo-level (fix failing test, small refactor with regression suite).
- Each task = `{repo fixture, request, hidden checks}`; fixtures are small
  synthetic-but-realistic repos (flask app, CLI tool, library) committed under
  `evals/fixtures/`. Hidden checks = pytest suites the agent never sees.
- Harness: run task → apply Forge config → score. Config = model ×
  effort × gate mask (each gate individually switchable — **this enables the
  ablation matrix**, so every gate needs an env kill-switch:
  `FORGE_GATE_SYNTAX=0`, `FORGE_GATE_ACTION=0`, …).

**3.2 Novel metrics** (computable from Forge's existing telemetry — this is
a genuine contribution; nobody measures agent honesty today):

| Metric | Definition |
| --- | --- |
| **TSR** task success rate | hidden checks pass after run |
| **ADT** act-don't-tell rate | action-shaped turns where ≥1 mutating tool succeeded / all action-shaped turns |
| **FVR** false-verification rate | turns claiming verification with zero commands run / turns claiming verification |
| **GER** grounded-edit rate | edits applied on tier-1/2 match / all attempted edits |
| **HIR** hallucinated-identifier rate | **shipped** — writes naming a module or symbol that does not exist / write attempts; the L2 rung decides this authoritatively at write time, so the metric counts its verdicts |
| **WCR** wasted-cycle rate | steps consumed by retry loops (same tool+args ≥2×) / total steps |
| Cost | wall-clock, tokens, steps per task |

**3.2b The detect/enforce separation (a design rule, learned the hard way).**
Every gate must keep *detecting* when it is configured not to *block*.
Implementing HIR exposed why: with `gate_resolution=0` the rung originally
stopped running, so the ablation that permits the most hallucinated imports
would have reported **HIR 0%** — the exact inverse of the truth, and a result
that would have survived review because it looks plausible. `RungResult`
therefore carries `enforced`, and an unenforced failure is still written to
the trace. Any future gate must follow the same rule or its ablation column
is meaningless.

**3.3 Statistical design.** ≥5 seeds per config (temperature variance),
report mean±CI, paired bootstrap for significance. Baselines: (1) raw
tool-loop with all gates off (same model — the critical ablation), (2) Aider
with the same Ollama model, (3) OpenHands with the same model if feasible.
Models: qwen2.5-coder:7b, qwen2.5-coder:14b (scale trend), deepseek-coder-v2-lite,
llama3.1-8b (non-code control).

Deliverable: `forge eval` producing a JSON report. **Every milestone below
gets its numbers from this.**

### 3.4 STATUS: shipped, and it immediately paid for itself

`forge eval [--tier N] [--seeds K] [--ablation NAME]` runs the suite, scores
with hidden pytest checks the agent never sees, and derives ADT/FVR/GER/WCR
from the trace. Six ablation presets are wired (`all-gates` … `no-gates`),
and every gate has an env kill-switch. Detection is deliberately decoupled
from correction: **detectors run even when a gate is disabled**, so an
ablation still measures the violations it is no longer preventing.

**First baseline** (qwen2.5-coder:7b, effort=smart, all gates, tier 1, 3
seeds = 9 runs):

| metric | value |
| --- | --- |
| task success rate | 66.7% (6/9) |
| act-don't-tell (ADT) | 66.7% |
| false-verification (FVR) | 20% (n=5) |
| grounded-edit (GER) | 66.7% |
| wasted-cycle (WCR) | 7.2% |
| tool reliability | 84.4% |

Three findings from building it — each one a reason the harness had to come
first:

1. **A measurement bug that would have corrupted the headline metric.** The
   false-verification *detector* (fine as a gate trigger, where a false
   positive costs one cheap nudge) was scoring honest disclaimers as lies:
   after being nudged, the model said *"I did not run any commands … please
   run the relevant tests"* — the phrase "run the relevant tests" matched.
   FVR read 100% when the true value was 0%. Fixed with sentence-level
   negation/deferral analysis (`claims_verification`). **Lesson: a gate
   trigger and a scientific instrument have different precision
   requirements; never reuse one as the other without re-validating.**
2. **The action detector was under-arming.** `"Make it return 0.0"` did not
   register as an action turn — the verb list had no `make` and no inflected
   forms (`changing`, `fixing`), so both the gate and ADT silently missed
   real requests. Replaced with verb *stems* + inflections, plus a question
   guard so `"how do I add …?"` still isn't treated as a change request.
3. **A live death-spiral with a real root cause** (see §3.5).

### 3.5 Case study: the escape-repair defect (found by the harness, now fixed)

On `t1-add-function` the model attempted **11 edits and landed 0**, burning
16 tool calls — the precise failure `edit_repair` exists to prevent. Root
cause: the model over-escaped its JSON tool arguments, so newlines arrived as
the two characters `\` + `n` (and regex backslashes arrived doubled). The
needle was therefore a *single line* that could never match a three-line
span; the fuzzy tier then suggested a one-line "correction", the model
re-sent the same broken form, and it looped.

Fix: a new repair tier decodes literal `\n`/`\t`/`\r`/`\\` and retries the
ladder — never inventing a location, since the decoded needle must still
match text that provably exists. Effect on the same task:

| | before | after |
| --- | --- | --- |
| tool calls (that task) | 16 (11 failed) | 2 (1 failed) |
| wasted-cycle | 75% | 0% |
| suite wall-clock | 85.1s | 47.9s |

**This is a paper-grade anecdote**: it shows the evaluation harness finding a
defect that months of interactive use did not, and it generalizes — JSON
over-escaping is a systematic small-model behaviour, so the repair belongs in
the contribution list, not the bug list.

### 3.5b The first ablation table (F1, tier 1)

qwen2.5-coder:7b, effort=smart, tier 1, 2 seeds → **6 runs per configuration**.
Each row disables exactly one mechanism; `no-gates` disables all of them.

| config | TSR | ADT ↑ | FVR ↓ | GER ↑ | WCR ↓ | tool calls | time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **all-gates** | 66.7% | 83% | **25%** | 83% | **0%** | 11 (1 failed) | 84s |
| no-syntax | 66.7% | 67% | 100% | 67% | 6% | 9 (2 failed) | 96s |
| no-edit-repair | 66.7% | 67% | 60% | **58%** | 12% | **17 (7 failed)** | 137s |
| no-action-gate | 66.7% | 80% | **100%** | 100% | **17%** | 15 (4 failed) | 696s |
| no-preflight | 66.7% | 100% | 33% | 100% | 10% | 15 (0 failed) | 91s |
| no-resolution | 66.7% | 67% | 80% | 67% | 8% | 11 (2 failed) | 112s |
| no-gates | **50.0%** | 67% | 80% | 75% | 6% | 12 (1 failed) | 79s |

**Read this table honestly.** TSR is 4/6 for every configuration except
`no-gates` at 3/6 — a one-task difference at n=6, which is **noise, not a
result**. (An earlier run of the same `all-gates` configuration scored 83.3%;
the variance is larger than the effect.) Nothing about task success can be
claimed from tier 1, and the paper must not try.

**What the table does show is mechanism-specific and interpretable** — each
gate degrades precisely the metric it was designed to protect:

- **`no-action-gate` → FVR 25% → 100%.** With the honesty gate off, *every*
  verification claim the model made was unbacked. It also produced the worst
  WCR (17%) and took **696s vs 84s** — with nothing to stop it, one run spun
  in a loop the gate would have broken.
- **`no-edit-repair` → GER 83% → 58%**, with tool failures rising 1 → 7 and
  calls 11 → 17. Removing edit repair returns the agent to the retry
  death-spiral, exactly as designed.
- **`all-gates` has the best FVR (25%) and the only 0% WCR** — the full stack
  is the most honest and least wasteful configuration, at moderate cost.
- **HIR was 0% everywhere and is therefore uninformative here**: tier-1 tasks
  are single-file and import nothing, so the resolution rung never had an
  opportunity to fire. HIR needs T2/T3 to mean anything — do not report it
  from tier 1.

**Conclusion for the paper.** At tier 1 the gates buy *honesty, groundedness
and efficiency*, not task success. The TSR claim must be earned at tier 2/3,
where a failed edit or a hallucinated import actually costs the run. That is
the next experiment, and it is an overnight job (see §5).

### 3.5c The full-suite sweep — a negative result, and what it means

All 3 tiers × 7 configurations × 3 seeds = 147 runs, qwen2.5-coder:7b,
effort=smart. **The headline is a benchmark failure, not an agent result.**

| config | TSR | T1 | T2 | T3 | ADT | FVR ↓ | HIR | WCR ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all-gates | 23.8% | 5/9 | **0/6** | 0/6 | 75% | **38%** | 0% | 12% |
| no-syntax | 28.6% | 6/9 | 0/6 | 0/6 | 86% | 43% | 0% | 11% |
| no-edit-repair | 28.6% | 6/9 | 0/6 | 0/6 | 76% | 36% | 0% | 11% |
| no-action-gate | 28.6% | 6/9 | 0/6 | 0/6 | 71% | **67%** | 0% | 6% |
| no-preflight | 33.3% | 6/9 | 0/6 | 1/6 | 95% | 33% | 0% | 7% |
| no-resolution | 28.6% | 5/9 | 0/6 | 1/6 | 75% | 53% | 0% | 13% |
| no-gates | 28.6% | 5/9 | 0/6 | 1/6 | 65% | 45% | 0% | 4% |

**T2 is 0/6 in every single configuration and T3 is at most 1/6.** That is a
floor effect, and it is exactly as uninformative as SWE-bench's ceiling
effect — the failure SWE-micro was built to avoid. A tier that no
configuration ever solves cannot discriminate between configurations, so
**no TSR claim can be made from this run either**. The benchmark, not the
agent, is what needs fixing first.

**Root cause of the T2 floor (diagnosed on `t2-wire-validator`).** The task
has two parts: add `validate_email()`, then make `register()` reject a bad
address. The model completes part 1 — 3 of the 4 hidden tests pass — and then
**drifts onto self-invented work** ("Step 3: add a test for validate_email")
instead of finishing part 2. The failure is not hallucination and not a bad
edit; it is **partial completion with task drift** on a multi-part
instruction. Every existing gate is blind to it: the code it wrote is
syntactically valid, resolves, and landed on real text.

**This names a gap in the contribution list.** Forge verifies that each
*action* is sound but never that the *request* was covered. The completeness
check added for `genius` effort is the only mechanism aimed at this, and the
sweep never exercised it (it ran at `smart`). Two consequences:

1. ~~Run the effort comparison first~~ **— done, and the hypothesis is
   refuted.** `genius` on tier 2 (2 seeds): **0/4, identical to `smart`.**
   The completeness check pushed ADT to 100% and WCR to 0% — the model always
   acted and never thrashed — and still solved nothing. A prompt-level
   "re-read the request" nudge does not produce requirement coverage. This is
   the project's recurring lesson applied to itself: **the fix has to be
   structural, not a stronger instruction.** C9 is therefore promoted from
   "nice to have" to the critical path, and it must be built as a real
   checklist the turn cannot end without satisfying, not as another nudge.
2. **Add a T1.5 tier**: two-part single-file tasks, to sit in the gap
   between "edit one function" (67% solved) and "coordinate two files" (0%).
   A benchmark needs a rung the system passes *sometimes*.

**HIR was 0% everywhere again**, even on cross-file tasks — because the runs
never got far enough to write a cross-module import. The metric remains
unexercised, and reporting it as "no hallucinations" would be dishonest.

**What did replicate:** the honesty effect. `all-gates` FVR 38% vs
`no-action-gate` 67%, the same direction and rough magnitude as tier 1. The
action gate's effect on truthfulness is now the one finding observed twice,
across different task difficulties.

### C9 — Requirement coverage verification (new, promoted by 3.5c)
Decompose the request into its atomic requirements before work starts, and
verify each one is satisfied before the turn may end — the missing rung that
checks the *request* rather than the *action*. The thinker model already
produces a structured brief (INTENT/STEPS/WHERE/VERIFY); the steps are a
requirement list waiting to be used as a checklist rather than a hint.

### 3.5d C9 built and measured: the blocker was not what it looked like

Building requirement coverage and running it on the new T2 tier produced
three results, none of them the expected one.

**1. The T2 blocker was a missing tool, not missing coverage.** The trace of
a failing run showed the model calling `edit_file` with an **empty
`old_string`** — its way of saying "put this new function at the end" — being
refused, then deflecting until the nudge budget ran out. `mutated: False`;
nothing was ever written. Adding a function to an existing file is the
commonest edit there is and Forge had no way to express it: `edit_file`
needs a unique anchor, `write_file` demands the whole file. `append_to_file`
now fills that gap, and the mechanical failures vanished:

| tier 2, 6 runs | before | after |
| --- | --- | --- |
| act-don't-tell | mixed | **100%** |
| tool reliability | 77-89% | **100%** (0 failed calls) |
| grounded-edit | 58-83% | **100%** |
| false-verification | 33-100% | **0%** |
| tasks solved | 0/6 | 0/6 |

Every mechanical failure mode is gone and **task success did not move**. That
separation is itself the finding: the remaining tier-2 gap is semantic — the
model acts cleanly and reliably on the wrong thing. Gates cannot fix wanting
the wrong outcome, and no further gate should be built expecting them to.

**2. The coverage gate works, and is not sufficient.** On `t2-add-and-use`
it decomposed the request into 4 requirements, correctly identified the
unmet one (*"Define a function `apply_discount(amount, percent)` in
prices.py"*), and sent the model back — twice. The model responded by running
`python -m unittest discover` four times and never writing the function. The
mechanism is sound; the model simply could not act on a correct, specific
instruction. This is the strongest evidence yet for the thesis' limit:
**verification tells you what is wrong, it cannot supply capability.**

**3. A metric validity bug, caught by that same trace.** The turn was
recorded as `mutated: True` while no file was ever written, because
`run_command` is flagged mutating (it needs permission) and ADT counted any
mutating tool. So **ADT was inflated across every result reported above**,
and the act-don't-tell gate failed to fire on a model that merely re-ran the
tests. ADT now counts only file-changing tools. This is the third measurement
bug the harness has caught in its own instruments — a pattern worth stating
in the paper: *instrument bugs and agent bugs are equally likely, and only
tracing real runs finds either.*

### 3.5e Bug-hunt pass: the bottleneck was Forge, not the model

A systematic hunt (adversarial tool probes, end-to-end API exercise, live
trace reading) found four defects. None was in the model.

| | before | after |
| --- | --- | --- |
| overall TSR | 23.8% | **35.0%** |
| T1 | 5/9 (56%) | **5/6 (83%)** |
| T2 | 0/6 | **2/6** |
| wasted-cycle | 12% | **2%** |
| tool reliability | 84% | **90%** |
| act-don't-tell | 75% | **85%** |

**The most expensive bug never showed up in a metric.** `Path.write_text`
defaults to `newline=None`, which rewrites every `
` as `os.linesep`. On
Windows every LF file Forge touched became CRLF, so editing one line of an
ordinary repository produced a diff touching *every* line — unreviewable, and
an instant merge conflict against every other checkout. It affected
write/edit/append, the ledger's backups and its undo. Reads normalise CRLF to
`
` for reliable matching, so writes now restore the file's own style.

The others: the syntax gate used `ast.parse`, which **accepts** `return`,
`break` and `yield` outside their block (those are symbol-table errors, raised
by `compile`) — a file Python refuses to run was passing; multi-line edit
replacements arrived at column zero and dedented code out of its function;
and a turn was bounded in steps but not in TIME, so one task ran 611 seconds.

**A design rule earned the hard way.** The dangling-reference check was first
wired per-write. A rename *must* pass through an inconsistent state — the
definition renamed, callers not yet — so blocking each write trapped the
agent: T2 fell 3/6 to 1/6 with 83% wasted cycles. Moved to a single turn-end
check, it recovers 3/6 and still catches the missed caller. **Never gate an
inherently multi-step operation at each step.**

**Where the ceiling now is.** T3/T4 remain 0. Mechanical failure is largely
gone — tool reliability 90%, wasted-cycle 2%, HIR 0% — so what is left is
semantic: the model acts cleanly on the wrong thing. That is the honest
boundary of the structural-verification thesis, and it is where C2
(execution-guided candidate search) has to earn its place.

### 3.6 What the first numbers already tell us

- The remaining tier-1 failure is **deflection, not capability**: ADT 0% on
  that task, the model pastes a correct-looking function instead of writing
  it. Honesty gates bounce it, but with a bounded nudge budget it can still
  end the turn having done nothing. **Next experiment:** does raising the
  nudge budget or escalating to `genius` convert those into successes?
- **Gates cost time and do not (yet) change tier-1 TSR** — the `no-gates`
  ablation also scored 66.7%, three times faster. Do not oversell: the
  honest framing is that gates buy *honesty and grounding* (FVR, GER, WCR)
  at tier 1, and the TSR case must be made at **tier 2/3**, where recovery
  from a bad edit actually matters. Running the full suite across all
  ablations × seeds is the immediate next job.
- Tier 1 sits at 67% — a responsive band, exactly as intended. SWE-bench
  would have returned ~0% and taught us nothing.

---

## 4. Ranked novel contributions (the roadmap)

Ordered by novelty × feasibility ÷ risk. Each: what, why novel, how, measure.

### C1 — The Verification Ladder (Milestone 18) ★ core of the paper
**STATUS: L1–L3 shipped** (`src/forge/verify/`). `Ladder.check(path, original,
new)` climbs syntax → resolution → types, stops at the first failure, and
returns that rung's own diagnostic. Wired into `write_file` and `edit_file`;
`FORGE_GATE_RESOLUTION=0` / `no-resolution` ablation isolate it.

**L2 (resolution) is the novel rung and it works**: a file can parse
perfectly and still import something that does not exist — the single most
characteristic code-model hallucination, invisible to every cheaper check.
L2 resolves each import against the stdlib, installed distributions, and the
repository, and for repo modules verifies the *imported names* exist,
offering a `did you mean` when a near-match is found. Verified end-to-end:
`from utils import make_magic` is refused with
`'utils' does not define 'make_magic'`, nothing is written, while
`from utils import helper` passes.

Two invariants keep it from trapping the agent (both inherited from the
syntax gate, both tested): only NEW failures block, and a failure returns the
rung's diagnostic rather than a generic refusal. Line-number drift is
normalised so a pre-existing bad import stays pre-existing after it moves.

Remaining: L4 (impacted-test selection via the import graph) and L5 (reviewer)
at the orchestrator level; measure TSR/HIR against rung depth.

Formalize the scattered gates into an explicit, ordered, cost-aware ladder
every mutation climbs before it counts as progress:

```
L0 schema      tool call parses against JSON schema        (~0 ms)
L1 syntax      AST/parser accepts the resulting file       (~10 ms)
L2 resolution  imports resolve, referenced symbols exist    (~50 ms)
L3 types       incremental type check of changed files      (~1-5 s)
L4 tests       impacted-test subset passes                  (~seconds)
L5 review      independent reviewer judges real diff        (~LLM call)
```

- New module `src/forge/verify/ladder.py`: `Ladder.check(change) ->
  VerdictReport` with per-rung results; write/edit tools call it instead of
  `syntax_check` directly. L2 uses the existing import graph + symbol index
  (`repo/graph.py`, `repo/symbols.py`) — cheap and already built. L3 shells
  to `pyright --outputjson` / `mypy --incremental` when available, skips
  otherwise. L4 needs impacted-test selection: map changed files → tests via
  the import graph (tests that transitively import the changed module).
- **Failure at rung N returns the rung's diagnostic to the model** — the
  grounding message pattern that already works for edits, generalized.
- Novelty: not any single check (all exist in CI) but the *in-loop,
  cost-ordered, diagnostic-feedback* formulation for agents + the ablation
  showing which rungs buy what for which model sizes. Reviewers can replicate.
- Measure: TSR/GER/HIR vs rungs enabled (L0-1 vs L0-2 vs L0-3 vs full).

### C2 — Execution-guided candidate search (Milestone 19) ★ test-time compute
When a mutation fails the ladder or the reviewer rejects, don't just retry —
**branch**: sample k candidate continuations (temperature-diverse, the
escalation already exists), apply each in an isolated *workspace overlay*,
climb the ladder on each, keep the best-scoring branch.

- `src/forge/verify/overlay.py`: copy-on-write overlay — materialize only
  touched files into `.forge/overlays/<branch>/`, run checks with
  `cwd=overlay` (Python import path juggling is the main risk; start with
  single-file tasks where it's trivial).
- `src/forge/orchestrator/search.py`: beam of width k=3, depth 2; score =
  highest rung reached, ties → fewest changed lines. Genius effort enables
  it; the eval measures success-vs-compute curves (the "scaling law" figure
  for structural compute on a fixed 7B — that figure alone is paper-worthy).
- Prior art to cite and differentiate: CodeT/AlphaCodium rerank *programs*
  against tests; Forge searches *editing trajectories* against a
  multi-rung verifier inside a live repo.

### C3 — Action-level speculative agency (Milestone 20) ★ most novel, riskiest
Speculative decoding, lifted from tokens to *actions*: a tiny draft model
(qwen2.5-coder:0.5b — already installed) proposes the next tool call; the 7B
only validates/overrides when the draft is risky or the ladder rejects it.

- Read-only, low-entropy calls (read_file, grep, list_dir on obvious targets)
  are where 7B spends most of its latency; a 0.5B drafts these nearly free.
- `src/forge/agents/speculative.py`: draft proposes call → acceptance policy
  (call is read-only AND schema-valid AND arguments reference existing paths
  → execute directly; else escalate to 7B with the draft as a hint).
- Measure: wall-clock speedup at equal TSR; 7B-invocations saved. Honest
  framing: if TSR drops >2pts, report the tradeoff — negative results on the
  acceptance-policy design are still contribution.
- Nothing published does draft/verify at tool-call granularity for coding
  agents. This is the "novel aspect" headline if it works.

### C4 — Verified Skill Library (Milestone 21) ★ self-improvement story
Execution memory currently stores *lessons* (text). Evolve it into *skills*:
parameterized, replayable tool-call macros distilled from successful
trajectories — **admitted to the library only if their replay passes the
ladder on a fresh fixture** (verification-gated admission is the delta over
Voyager-style libraries, which have no correctness pressure).

- `src/forge/memory/skills.py`: after an approved task, the thinker model
  abstracts the trajectory (concrete paths → parameters) into
  `{name, preconditions, steps, verification}`; a replay harness re-runs it
  on a synthetic variation; only passing skills persist. Retrieval: BM25 over
  skill descriptions at planning time; the planner may emit `use_skill(...)`.
- Measure: TSR and WCR on task families over repeated exposure (does Forge
  get *faster and more reliable* at the 5th "add a CLI flag" than the 1st?)
  — the learning-curve figure.

### C5 — Predictive blast-radius verification (Milestone 22)
Before an edit, the agent must *predict* impact ("this change affects
files X,Y; tests T1,T2 should still pass"); after, the system *diffs
prediction against reality* (import graph + actually-failing tests).
Mismatch = grounding failure signal fed back like an edit-repair message.

- Cheap: the import graph exists; prediction is one structured call
  (`predicted_files`, `predicted_tests` added to edit-intent); comparison is
  set arithmetic. Novelty: "predict-then-verify" as a *first-class gate* on
  agent world-modeling, plus a new metric (**BPA** blast-prediction accuracy)
  correlating world-model quality with task success across model sizes.

### C6 — Adversarial regression memory (Milestone 23)
Every reviewer rejection or ladder failure becomes a *permanent artifact*:
auto-generate a minimal failing check capturing the mistake, store per-repo
(`.forge/regressions/`), and run it forever after as an L4 rung member.
The repo accumulates an immune system from the agent's own failures.
Measure: repeat-failure rate over time; distinct-vs-repeated error classes.

### C7 — Introspective escalation (Milestone 24)
Use token-level uncertainty (Ollama exposes logprobs) at *decision points*:
high-entropy tool-call generations trigger escalation — best-of-N (C2),
thinker consult, or an honest "I'm not sure" to the user — instead of
confidently wrong action. Effort levels become a *dynamic* policy rather
than a static switch. Measure: selective-prediction curves (risk-coverage),
ECE of the escalation trigger vs task failure.

### C8 — Paper-grade packaging of the honesty gates (with M17's numbers)
The action gate + false-verification gate + their measured ADT/FVR deltas
is a small, clean, standalone contribution ("Honesty Gates: enforcing
act-don't-tell in small-model agents") — a workshop paper that can ship
*early* while the big system paper matures. Cheap insurance.

---

## 5. Suggested execution order & timeline

| Phase | Milestones | Output |
| --- | --- | --- |
| ~~1~~ ✅ | M17 harness + gate kill-switches | **done** — see §3.4; baselines + two instrument fixes + the escape-repair contribution |
| ~~1b~~ ✅ | tier-1 ablation table, 7 configs × 6 runs | **done** — see §3.5b; mechanism-specific effects confirmed, TSR within noise |
| 1c (next) | **tier 2/3 × all ablations × 5 seeds — overnight run**; this is where TSR and HIR become measurable | the paper's core figure (F1) |
| 2 (weeks 3-5) | M18 ladder (L2/L3/L4 rungs) | core ablation table; C8 workshop draft possible here |
| 3 (weeks 5-8) | M19 overlay search | compute-vs-success curves |
| 4 (weeks 8-11) | M20 speculative agency | latency results (or honest negative result) |
| 5 (weeks 11-14) | M21 skills or C5 blast-radius (pick by phase-2 findings) | learning curves / BPA |
| 6 (weeks 14-16) | writing, ablation reruns, artifact packaging (Docker + fixtures + seeds for replication badge) | submission |

Venues (in order of fit): **ASE** (IEEE/ACM, agent systems fit), **ICSE**,
**FSE**, journal track **IEEE TSE**; early/partial results → ASE/ICSE
workshops (LLM4Code, AIware) or **IEEE Software** (practitioner angle:
"trustworthy local AI engineers"). ArXiv preprint as soon as Phase 2 numbers exist.

---

## 6. Paper skeleton (draft when Phase 2 lands)

- **Title candidates:** "Verifier-Gated Autonomy: Reliable Software-Engineering
  Agents on Consumer Hardware" / "Structure over Scale: A Verification Ladder
  for Small-Model Coding Agents."
- **Abstract shape:** small local models fail as agents not from lack of
  knowledge but from ungrounded action (quantify with ADT/FVR/HIR of the
  gate-free baseline) → we introduce the verification ladder + honesty gates +
  execution-guided search → TSR ×N improvement on SWE-micro at 7B, all
  data on-device → structure substitutes for scale up to tier X tasks.
- **Claims → evidence:** every claim in the intro must map to one figure:
  (F1) ablation table gates×metrics, (F2) ladder-depth vs TSR, (F3)
  compute-vs-success curve, (F4) honesty metrics before/after, (F5)
  cross-model scale trend, (F6) qualitative trajectory case study.
- **Threats to validity to pre-empt:** benchmark self-construction (mitigate:
  release publicly, third-party tasks in T3), fixture leakage into model
  pretraining (use post-cutoff synthetic repos), single-language scope
  (state it; the ladder is language-parametric by design).

## 7. Deliberately out of scope (don't get distracted)

Fine-tuning/LoRA (different paper, needs GPUs), multi-user/cloud (kills the
local thesis), more UI polish (product, not research), RAG over external
docs (crowded field, low delta), model routing across providers (engineering).

---

*Everything here builds on interfaces that already exist: gates hook the
tool layer, search hooks the orchestrator, skills hook ExecutionMemory,
metrics hook the Recorder. No rewrites required — that modularity was the
point of milestones 1-16.*
