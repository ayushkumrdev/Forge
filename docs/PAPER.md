# Scaffold, Not Scale: Diff-Aware Structural Verification for Autonomous Coding on a 7B Model

**Draft — Forge project. Numbers in §6 are measured; every claim traces to a
recorded run. Placeholders are marked TODO and must not be filled by
estimation.**

---

## Abstract

Autonomous coding agents are usually improved by using a larger model. We ask
what remains when that lever is unavailable. Forge is a local agent running
`qwen2.5-coder:7b` on consumer hardware, built around a verification ladder
that checks each change against the repository rather than against the
model's confidence. Over one development campaign we traced every failure of
the agent on a purpose-built benchmark and attributed each to its true cause.
**Of thirty-one defects found, thirty were in the scaffold — the harness, the
tools, or the verification machinery — and one was a limitation of the
model.** Task success on our benchmark rose from 23.8% to 56.7% without
changing the model, its weights, or its context budget. Two tasks that had
never once succeeded reached 3/3 and 1/4.

We report three results we believe generalise. First, **diff-aware
verification** — checks that are wrong only relative to the prior state of the
repository — catches a class of defect no static analysis of the finished
file can see, and no prior agent verification scheme we are aware of
implements. Second, **execution scheduling** matters as much as verification:
the same model, given the same instruction, succeeds or fails depending on
*when* its attention is spent, and decomposition helps for independent
requirements while actively destroying work that is one coherent artifact.
Third, and most uncomfortably, **two of the most expensive defects we found
were our own verification machinery misfiring on work it did not
understand** — a failure mode that any paper advocating verification-as-
capability has an obligation to report.

---

## 1. Introduction

The dominant response to a coding agent that fails is a bigger model. That
response is unavailable to anyone running locally on fixed hardware, and it
is uninformative even when available, because it does not say what the agent
was actually getting wrong.

This paper takes the opposite approach. We fixed the model — a 7B parameter
code model on a consumer GPU — and asked how far the surrounding system can
be pushed. The method is deliberately unglamorous: run the agent on a
benchmark, read the execution trace of every failure, find the cause, fix it,
and measure again.

The result that surprised us is the ratio. We expected to find that a 7B
model has a hard ceiling and to characterise it. Instead, **every time we
concluded "this is the model's limit", tracing proved otherwise.** Thirty-one
defects, thirty of them ours.

### 1.1 Contributions

1. **Diff-aware structural verification** (§3.2): a family of checks that
   compare a change against the state before it, catching defects that are
   invisible in the finished file.
2. **A defect taxonomy from trace archaeology** (§5, Table 3): every failure
   of a local agent over one campaign, classified by true cause.
3. **The scheduling result** (§6.3): decomposition helps for independent
   requirements and harms coherent artifacts — with the boundary measured.
4. **Two negative results about verification itself** (§6.4): checks that
   caused the failures they then detected.
5. **SWE-micro** (§4): a tiered benchmark where every task carries a verified
   reference solution, so an unsolvable task cannot be mistaken for an agent
   failure — a mistake we made and which cost us weeks.

---

## 2. Related work

**Scaffold minimalism.** `mini-swe-agent` reduces an agent to roughly 100
lines with no custom tool interfaces and scores above 74% on SWE-bench
Verified, matching far more elaborate frameworks, and its authors conclude
the bottleneck is the model rather than the scaffold. This is the strongest
challenge to our position and we accept it *for frontier models*. Our claim
is a conditional one, and we state it as such:

> **The value of scaffolding is inversely proportional to model capability.**
> With a strong model, verification machinery is overhead. At 7B it is the
> capability.

We propose the experiment that settles this in §9: the same ablation matrix
at two model sizes, locating the crossover per gate. We are not aware of that
experiment having been run.

**Tool-integrated verification.** T1 [Kim et al., arXiv:2504.04718] shows
small models fail at verification requiring memorisation and that offloading
it to tools lets a 1B model outperform an 8B. This is our thesis in a
different domain, and it is the closest prior art. **T1 verifies answers;
Forge verifies changes.** The distinction is what makes §3.2 novel: a
narrowed function signature, a call left dangling by a rename, or a package
that exports nothing are all *correct in isolation* and wrong only relative
to what preceded them. Answer verification has no analogue for this.

**Repository generation.** NL2Repo-Bench [arXiv:2512.12730] identifies the
dominant failure modes of building a repository from a specification:
premature termination, loss of global coherence, fragile cross-file
dependencies, and inadequate planning, noting that "agents often struggle to
correctly structure the package (e.g. missing `__init__.py`) or manage
internal dependencies". We independently rediscovered exactly this defect
from a trace (§6.4) before finding their paper; our §3.2 check for it is, to
our knowledge, the first mechanical detector reported. They also find
`task_tracker` usage correlates 0.711 with success — a *persistent* planning
artifact — which we use in §6.3 to explain why our own decomposition helps in
one regime and harms in another.

**Design contracts.** CodeTeam [arXiv:2606.22082] proposes a software design
sketch used as a machine-checkable contract over file ownership, public
interfaces and dependencies. Our §9 proposal for greenfield work is this idea
applied to the failure mode our traces identify as dominant.

**Context management.** Context-Folding [arXiv:2510.11967] formalises
branching into a sub-trajectory and folding it back as a summary, reporting
parity with ReAct at 10× smaller active context. Our focused pass (§3.3) is
an unlearned instance of the same shape, driven by verification signals
rather than a trained reward. Notably they *fold a summary back*; we discard,
which §6.3 shows is the wrong choice when coherence matters.

**A caution we adopt.** NL2Repo-Bench reports that reasoning-first models
underperform, describing "a self-reinforcing echo chamber where the model
convinces itself of correctness", with a 49% early-stop rate. We shipped a
self-briefing stage late in this campaign and, on that evidence, treat it as
**unvalidated and suspect** rather than as an improvement (§8).

---

## 3. Forge

### 3.1 The verification ladder

Every write climbs an ordered set of checks, cheapest first, stopping at the
first failure and reporting *that* check's own diagnostic so the model can
act on it. Only **new** failures block: a file that was already broken is
never blamed on the current change, so the agent can work inside a
half-finished refactor.

| rung | question | cost |
| --- | --- | --- |
| L1 syntax | does it parse | ~10 ms |
| L2 resolution | do the names it references exist | ~50 ms |
| L3 types | does a type checker accept it | seconds |
| L4 runtime | **does it import** | ~200 ms |

L4 is the rung that stops reading and runs. Importing is the cheapest
possible execution — no test to author, no model judgement, no output to
interpret — and it catches what no static rung can (§3.2).

### 3.2 Diff-aware checks (contribution 1)

These run **once at the end of a turn, never per write.** A rename must pass
through a state where a caller is broken; a package must pass through a state
where its `__init__.py` names a module not yet written. Gating each step
traps the agent — measured, twice, at a cost of 3/6 → 1/6.

| check | what it catches | invisible to |
| --- | --- | --- |
| dangling reference | rename updated the definition, missed a caller | L1, L2, L3 |
| narrowed signature | `register(name, email)` shipped as `register(user)` | all static |
| undefined call | the call landed, the import did not | L1, L3 |
| runaway recursion | `self._items.pop(0)` hand-edited to `self.dequeue()` | all static |
| non-boolean predicate | `return local and domain` yields `''`, not `False` | all static |
| unexported package | `__init__.py` exports nothing its modules define | all static |

Each was written after observing it happen, and each is validated against
this repository's own source and history before being trusted — a check that
fires on correct code is worse than no check, because it teaches the model to
ignore the channel. The narrowed-signature check was replayed over 20 commits
of real history with zero false positives; the boolean check was silent
across the entire source tree after one refinement, and the one hit it
produced before that refinement was a genuine false positive we fixed.

### 3.3 Focused passes and plan-first

A requirement executed in a clean context — nothing in front of the model but
that one instruction — succeeds where the same instruction ignored at the end
of a twenty-message conversation fails. Forge decomposes a multi-part request
up front and runs each requirement this way. §6.3 gives the boundary
condition, which is not what we expected.

### 3.4 Honesty gates

Detection always runs even when a gate is disabled, so an ablation still
measures the violation it stopped correcting. Metrics: act-don't-tell (ADT),
false verification (FVR), grounded edit (GER), wasted cycle (WCR),
hallucinated identifier (HIR), and empty-step rate (ESR, §6.5).

---

## 4. SWE-micro

SWE-bench is uninformative at this model size: a 7B general code model scores
near zero, and a floor tells you nothing about which mechanism helped. We
built a tiered benchmark where the agent operates on real fixture
repositories scored by hidden test suites it never sees.

| tier | shape |
| --- | --- |
| T1 | one edit in one file |
| T2 | several requirements, one file |
| T3 | cross-file |
| T4 | repo-level |
| T5 | greenfield — an empty directory |

**Every task carries a verified reference solution**, and a test applies each
solution and asserts the hidden checks pass. This is not ceremony. One task
was unsolvable for weeks because its hidden check imported the wrong module,
and every 0/3 we attributed to the agent was our bug. Any benchmark used to
diagnose an agent must prove its tasks are solvable.

T5 exists because "build me a project" is what users ask for and no tier
measured it. It is the only tier where the agent starts with no file tree to
ground against.

---

## 5. Method: defect archaeology

For each failure: read the execution trace, identify the true cause, classify
it as model or scaffold, fix it, re-measure. Traces are preserved per seed —
a change we had to make after two diagnoses were blocked by a harness that
overwrote the evidence.

**The methodological finding is that the trace, not the score, carries the
diagnosis.** Three of the last five defects were mechanisms that never ran: a
gate that was not armed, a retry whose condition could not be true, a step
discarded before it acted. Task success rate was identical before and after
one of the most damaging regressions we introduced.

---

## 6. Results

### 6.1 Aggregate

Measured over 3 seeds, 30 runs, tiers 1–4 (all-gates configuration,
`qwen2.5-coder:7b`):

| metric | value |
| --- | --- |
| task success rate | **56.7%** (17/30), from 23.8% at campaign start |
| tier 1 | 8/9 (88.9%) |
| tier 2 | 5/9 (55.6%) |
| tier 3 | 3/6 (50.0%) |
| tier 4 | 1/6 (16.7%) |
| act-don't-tell | 100.0% (n=30) |
| hallucinated identifier | 0.0% (n=22) |
| tool reliability | 87.0% (n=30) |
| grounded edit | 60.9% (n=22) |
| wasted cycle | 22.1% (n=30) |
| **false verification** | **64.7%** (n=17) |

TODO — re-measure across tiers 1–5 after the final set of changes; the run
above predates L4, the greenfield fixes and enforced verification.

### 6.2 Per-task, the two that never passed

| task | before | after | trace at success |
| --- | --- | --- | --- |
| t2-rename-in-file | 0/3, every run | **3/3** | 2 tool calls, ~25 s |
| t3-wire-validator | 0/3, every run | 1/3 | — |
| t5-build-package | 0/2 | 1/4 | 3 tool calls, 35 s |

`t2-rename-in-file` at 3/3 is one `rename_symbol` per name with no wasted
calls — the ideal trace for that task. Nothing about the model changed.

### 6.3 The scheduling result (contribution 3)

Plan-first — decompose up front, execute each requirement in a clean context
— took `t2-rename-in-file` from 0/3 to 3/3. On greenfield it was **actively
destructive**. The decomposer split one small package into six requirements
("a package is created", "`__init__.py` exists"), each executed in isolation,
and they fought: one created a stray `__init__.py` at the repository root,
another appended a bare identifier as a line of code. The artifact produced:

```python
__all__ = []

from . import primes_up_to
```

**A directory, its module and its exports are facets of one artifact, not
independent outcomes.** Isolating them destroys the coherence authoring
requires. This is the first mechanism in this project we had to *limit*
rather than extend, and it reconciles with NL2Repo-Bench's finding that a
*persistent* task tracker correlates with success: the benefit is planning,
the harm is fragmentation, and the two are separable.

Greenfield after disabling decomposition on an empty workspace, plus the
fixes in §6.4:

| | TSR | tool reliability | wasted cycle | tool calls |
| --- | --- | --- | --- | --- |
| baseline | 0/2 | 50% | 34% | 44 |
| + grounded errors, no split | 0/4 | 71% | 6% | 17 |
| + forward reference | **1/4** | **86%** | 8% | 16 |

### 6.4 Two negative results about verification (contribution 4)

**Our L2 rung caused the failure our new L4 check detected.** Building a
package, the model wrote:

```
write mathkit/__init__.py ""                          ok
write "from .primes import is_prime, primes_up_to"    REJECTED
write "from primes import is_prime, primes_up_to"     REJECTED
write "from . import primes"                          ok
write mathkit/primes.py                               ok
```

The rejected line was correct — `primes.py` was one write away. L2 asks
whether an import resolves *now*, and a package is built over several writes.
Refused twice, the model settled for `from . import primes`, which exports
nothing, producing precisely the empty-package defect our new check then
reported. We relaxed L2 to permit a forward reference from a new package's
own `__init__.py` to a direct sibling, while the package holds no other
modules; L4 catches what does not materialise.

**A prompt example is an instruction.** We added guidance pointing renames at
the correct tool and illustrated it with a bare arguments object rather than
a complete call. The model reproduced that shape as prose, inline-call
recovery could not parse it without a name field, and steps ended having done
nothing. Cost: tool calls per run 3–4 → 1, tool reliability 92% → 75%, **task
success rate unchanged**, which is why it went unnoticed.

### 6.5 A metric that catches what TSR hides (ESR)

Empty-step rate — the share of focused steps that changed nothing — was added
after a regression scored 100% tool reliability, 0% wasted cycles and 100%
act-don't-tell while doing half the work it was asked for. ESR read 61% on
that run and 5% after the fix. **A behavioural metric suite is not
decoration; it is the only thing that separates a clean run from a run that
barely happened.**

---

## 7. Findings we believe generalise

1. **An instruction to the model is a hope; a filter on its output is a
   guarantee.** Three prompt-level rules in this campaign lost to mechanical
   ones covering the same case, including one written an hour earlier.
2. **Never gate an inherently multi-step operation at each step.** Paid for
   three times: renames, package construction, and per-write dangling
   references (3/6 → 1/6).
3. **A per-step check must be computed from the step**, never from cumulative
   state earlier steps have written into. A skipped rename looked satisfied
   because a previous step had touched the same file.
4. **The first message of a step is not a verdict.** A 7B asked to do one
   thing frequently narrates before acting; a loop that treats narration as
   termination discards steps that were about to succeed. Every first attempt
   in one run was discarded this way.
5. **When a mechanism appears not to work, check that it ran.** Three of the
   last five defects were mechanisms that were never reached.
6. **Verify a new check against real code before trusting it.** Ours are run
   over this repository's own source and its commit history; a check that
   fires on correct code trains the model to ignore the channel.

---

## 8. Threats to validity

- **Scale.** 10 tasks, 3 seeds, one model, one language. Individual per-task
  deltas at n=3 are not significant; the aggregate and the mechanism-level
  traces carry the argument, not any single cell.
- **Benchmark authorship.** We wrote the benchmark we are measured on.
  Mitigations: hidden checks, verified reference solutions, a test asserting
  every task fails on the untouched fixture. It remains a real threat and the
  external calibration in §9 is required.
- **Ablations are stale.** The gate matrix has not been re-run since most of
  the mechanisms in §3 landed. No causal claim for any individual gate is
  made here.
- **`gate_intent_brief` is unvalidated** and the literature predicts it
  hurts (§2). It ships behind a flag and is not counted as a contribution.
- **FVR at 64.7% is the honest headline weakness.** Forge claimed
  verification it had not performed in two of three turns that claimed
  anything, including one reply that fabricated a terminal transcript. The
  fix — running the checks when the model claims they ran — is implemented
  but **not yet measured**.

---

## 9. Future work

1. **The crossover experiment.** The ablation matrix at two model sizes,
   locating per gate where scaffolding stops paying. This settles the
   disagreement with scaffold minimalism (§2) and is the strongest single
   experiment available to us.
2. **Interface contracts for greenfield.** A machine-checkable manifest of
   files, the public names each must define, and who imports whom — verified
   against the built tree. Targets the failure mode NL2Repo-Bench identifies
   as dominant and which our traces confirm.
3. **Fold step summaries back** into the main trajectory rather than
   discarding them, recovering coherence at Context-Folding's token cost.
4. **External calibration** on a public benchmark subset, to bound the
   benchmark-authorship threat.

---

## Appendix A — reproduction

```bash
forge eval --seeds 3                      # full suite, all tiers
forge eval --tier 5 --seeds 2             # greenfield only
forge sweep                               # ablation matrix
```

Traces are written per seed to
`.forge/evals/seed-<n>/<task>/.forge/logs/`, and every result in §6 is
reconstructible from them.
