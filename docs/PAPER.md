# Scaffold, Not Scale: What Actually Limits a 7B Coding Agent

**Working draft. Every number in this paper comes from a recorded run in
this repository. Nothing is estimated, and no figure has been replaced after
the fact by a more flattering one.**

---

## Abstract

When an autonomous coding agent fails, the usual remedy is a bigger model.
That remedy is unavailable to anyone running locally on fixed hardware, and
it is uninformative even when it works, because it never says what the agent
was getting wrong. We fixed the model instead. Forge runs qwen2.5-coder:7b on
a consumer GPU behind a verification ladder that checks each change against
the repository rather than against the model's confidence.

Over one development campaign we read the execution trace of every failure on
a purpose built benchmark and attributed each one to its true cause. Of
thirty one defects, thirty were in the scaffold and one was a limitation of
the model. Task success rose from 23.8 to 56.7 percent with the model, its
weights and its context budget untouched. Two tasks that had never once
succeeded reached 3/3 and 1/4.

On five real SWE-bench Lite instances the same system produced a well formed,
cleanly applying patch every time and resolved none of them, which locates
the boundary precisely: verification makes a change sound, and it does
nothing to find the right place to make it.

Three findings look transferable. Diff aware verification, meaning checks
that are wrong only relative to the previous state of the repository, catches
a class of defect that no analysis of the finished file can see. Execution
scheduling matters as much as verification does, since the same model given
the same instruction succeeds or fails depending on when its attention is
spent. And two of the most expensive defects we found were our own
verification machinery misfiring on work it did not understand, which is a
result that a paper advocating verification has an obligation to report.

---

## 1. Introduction

Ask why a coding agent failed and the answer usually offered is that the
model was too small. That answer is hard to falsify and it forecloses the
more useful question, which is what the agent actually did wrong.

This paper takes the opposite route. We held the model constant, a seven
billion parameter code model on a consumer GPU, and asked how far the
surrounding system could be pushed. The method is unglamorous. Run the agent
on a benchmark, read the trace of every failure, find the cause, fix it, and
measure again.

What surprised us was the ratio. We expected to characterise a hard ceiling
at 7B. Instead, every time we concluded that we had reached the model's
limit, tracing showed otherwise. Thirty one defects, thirty of them ours.

### 1.1 Contributions

1. Diff aware structural verification (section 3.2), a family of checks that
   compare a change against the state before it and catch defects that are
   invisible in the finished file.
2. A defect taxonomy built from trace archaeology (section 5), covering every
   failure of a local agent over one campaign, classified by true cause.
3. The scheduling result (section 6.3): decomposition helps when requirements
   are independent and harms coherent artifacts, with the boundary measured.
4. Three negative results (sections 6.4 and 6.5), including a mechanism that
   the literature predicted would hurt and which our own harness confirmed
   hurt.
5. SWE-micro (section 4), a tiered benchmark in which every task carries a
   verified reference solution, so an unsolvable task cannot be mistaken for
   an agent failure. We made exactly that mistake and it cost us weeks.
6. A measured boundary (section 6.7): on real SWE-bench instances the
   scaffold produces well formed patches every time and resolves nothing,
   because the binding constraint there is localisation rather than
   soundness, and nothing in section 3 addresses localisation.

---

## 2. Related work

### Scaffold minimalism

The strongest challenge to our position comes from mini-swe-agent, which
reduces an agent to roughly a hundred lines with no custom tool interfaces
and scores above 74 percent on SWE-bench Verified, matching much more
elaborate frameworks. Its authors conclude that the bottleneck is the model
rather than the scaffold.

We accept that conclusion for frontier models and state our own claim as a
conditional:

> The value of scaffolding is inversely proportional to model capability.
> With a strong model, verification machinery is overhead. At 7B it is the
> capability.

Section 9 proposes the experiment that settles this, which is the same
ablation matrix run at several model sizes to locate the crossover for each
mechanism. We are not aware of that experiment having been run.

### Tool integrated verification

T1 (arXiv:2504.04718) shows that small models fail at verification requiring
memorisation, and that moving it into tools lets a 1B model outperform an 8B.
This is our thesis in a different domain and it is the closest prior work.
The difference that makes section 3.2 novel is that T1 verifies answers while
Forge verifies changes. A narrowed function signature, a call left dangling
by a rename, or a package that exports nothing are all perfectly correct in
isolation and wrong only relative to what preceded them. Answer verification
has no analogue for this.

### Repository generation

NL2Repo-Bench (arXiv:2512.12730) identifies the dominant failure modes when
an agent builds a repository from a specification: premature termination,
loss of global coherence, fragile cross file dependencies, and inadequate
planning. It notes that agents often fail to structure a package correctly,
for instance by omitting `__init__.py`, or to manage dependencies between
their own modules.

We rediscovered precisely that defect from a trace before finding their
paper, and the mechanical detector in section 3.2 is, as far as we know, the
first one reported for it. They also find that use of a task tracker
correlates at 0.711 with success, which is a persistent planning artifact
rather than a transient one. Section 6.3 uses that distinction to explain why
our own decomposition helps in one regime and hurts in another.

### Design contracts

CodeTeam (arXiv:2606.22082) proposes a software design sketch used as a
machine checkable contract over file ownership, public interfaces and
dependencies. Our proposal in section 9 applies that idea to the failure mode
our traces identify as dominant.

### Context management

Context-Folding (arXiv:2510.11967) formalises branching into a sub trajectory
and folding it back as a summary, reporting parity with ReAct at a tenth of
the active context. Our focused pass in section 3.3 is an unlearned instance
of the same shape, driven by verification signals rather than a trained
reward. They fold a summary back where we discard, and section 6.3 shows that
discarding is the wrong choice when coherence matters.

### A caution we adopted, and then confirmed

NL2Repo-Bench reports that reasoning first models underperform, describing a
self reinforcing echo chamber in which the model convinces itself of
correctness, with a 49 percent early stop rate. We had shipped a self
briefing stage shortly before finding this. Rather than assume it helped, we
ablated it. Section 6.5 reports what happened.

---

## 3. Forge

### 3.1 The verification ladder

Every write climbs an ordered set of checks, cheapest first, stopping at the
first failure and reporting that check's own diagnostic so the model can act
on it. Only new failures block. A file that was already broken is never
blamed on the current change, so the agent can still work inside a half
finished refactor.

| rung | question | cost |
| --- | --- | --- |
| L1 syntax | does it parse | about 10 ms |
| L2 resolution | do the names it references exist | about 50 ms |
| L3 types | does a type checker accept it | seconds |
| L4 runtime | does it import | about 200 ms |

L4 is the rung that stops reading and runs. Importing a module is the
cheapest execution available. There is no test to write, no model judgement,
and no output to interpret, because the interpreter either loads the module
or says exactly why it cannot.

### 3.2 Diff aware checks

These run once at the end of a turn and never per write. A rename has to pass
through a state where some caller is broken. A package has to pass through a
state where its `__init__.py` names a module that has not been written yet.
Gating each step traps the agent, which we measured twice at a cost of 3/6
falling to 1/6.

| check | what it catches | invisible to |
| --- | --- | --- |
| dangling reference | rename updated the definition and missed a caller | L1, L2, L3 |
| narrowed signature | `register(name, email)` shipped as `register(user)` | all static rungs |
| undefined call | the call landed and the import did not | L1, L3 |
| runaway recursion | `self._items.pop(0)` hand edited into `self.dequeue()` | all static rungs |
| non boolean predicate | `return local and domain` yields `''` rather than False | all static rungs |
| unexported package | `__init__.py` exports nothing that its modules define | all static rungs |

Each of these was written after watching it happen. Each is validated against
this repository's own source and history before being trusted, because a
check that fires on correct code is worse than no check at all. It teaches
the model to ignore the channel. The narrowed signature check was replayed
over twenty commits of real history with no false positives. The boolean
check was silent across the whole source tree after one refinement, and the
single hit it produced before that refinement was a genuine false positive
that we fixed.

### 3.3 Focused passes and plan first execution

A requirement executed in a clean context, with nothing in front of the model
but that one instruction, succeeds where the same instruction ignored at the
end of a twenty message conversation fails. Forge decomposes a multi part
request up front and runs each requirement this way. Section 6.3 gives the
boundary condition, which was not what we expected.

### 3.4 Honesty gates

Detection always runs, even when a gate is switched off, so an ablation still
measures the violation it stopped correcting. The metrics are act don't tell
(ADT), false verification (FVR), grounded edit (GER), wasted cycle (WCR),
hallucinated identifier (HIR) and empty step rate (ESR, section 6.6).

---

## 4. SWE-micro

SWE-bench is uninformative at this model size. A 7B general code model scores
near zero, and a floor says nothing about which mechanism helped. We built a
tiered benchmark in which the agent works on real fixture repositories scored
by hidden test suites it never sees.

| tier | shape |
| --- | --- |
| T1 | one edit in one file |
| T2 | several requirements in one file |
| T3 | cross file |
| T4 | repository level |
| T5 | greenfield, an empty directory |

Every task carries a verified reference solution, and a test applies each
solution and asserts that the hidden checks pass. This is not ceremony. One
task was unsolvable for weeks because its hidden check imported the wrong
module, and every 0/3 we had attributed to the agent was our own bug. A
benchmark used to diagnose an agent has to prove its tasks are solvable.

T5 exists because building a project is what users actually ask for and no
tier measured it. It is the only tier where the agent starts with no file
tree to ground against.

Section 9 covers external calibration on SWE-bench itself, which is required
to bound the obvious threat that we wrote the benchmark we are measured on.

---

## 5. Method

For each failure we read the execution trace, identified the true cause,
classified it as model or scaffold, fixed it, and measured again. Traces are
kept per seed, a change we had to make after two diagnoses were blocked by a
harness that overwrote its own evidence.

The methodological finding is that the trace carries the diagnosis and the
score does not. Three of the last five defects were mechanisms that never
ran: a gate that was not armed, a retry whose condition could not be true,
and a step discarded before it acted. Task success rate was identical before
and after one of the most damaging regressions we introduced.

---

## 6. Results

### 6.1 Aggregate

Measured over three seeds and thirty six runs across all five tiers, using
qwen2.5-coder:7b with all gates enabled:

| metric | value |
| --- | --- |
| task success rate | 44.4 percent (16/36) |
| tier 1 | 9/9 |
| tier 2 | 3/9 |
| tier 3 | 2/6 |
| tier 4 | 1/6 |
| tier 5 | 1/6 |
| act don't tell | 94.3 percent (n=35) |
| false verification | 47.6 percent (n=21) |
| grounded edit | 54.9 percent (n=24) |
| wasted cycle | 22.5 percent (n=36) |
| tool reliability | 81.9 percent (n=36) |
| hallucinated identifier | 0.0 percent (n=27) |
| empty step | 19.0 percent (n=22) |

The headline figure is not comparable to the 56.7 percent reported earlier in
the campaign, because tier 5 was added and it is the hardest tier. On tiers 1
to 4 alone the same run gives 15/30 against 17/30 previously, which is a
regression of two tasks. Section 6.5 identifies the cause and reports what
happened when we removed it.

This run had self briefing enabled, which section 6.5 shows was a mistake, so
the aggregate above measures a configuration we no longer ship. The targeted
ablation in section 6.5 is the trustworthy figure for the affected tasks, and
a full suite run under the shipped configuration is outstanding. We leave the
flawed number in place rather than quietly replacing it, because the sequence
of measurements is itself the argument of section 5.

### 6.2 The two tasks that had never passed

| task | before | after | trace at success |
| --- | --- | --- | --- |
| t2-rename-in-file | 0/3 on every run | 3/3 | two tool calls, about 25 seconds |
| t3-wire-validator | 0/3 on every run | 2/3 | see section 6.5 |
| t5-build-package | 0/2 | 1/4 | three tool calls, 35 seconds |

The passing trace for t2-rename-in-file is one `rename_symbol` call per name
with nothing wasted, which is the ideal trace for that task. Nothing about
the model changed to produce it.

### 6.3 The scheduling result

Plan first execution, meaning decompose up front and run each requirement in
a clean context, took t2-rename-in-file from 0/3 to 3/3. On greenfield work
it was actively destructive. The decomposer split one small package into six
requirements, among them "a package is created" and "`__init__.py` exists",
each executed in isolation, and they fought each other. One step created a
stray `__init__.py` at the repository root. Another appended a bare
identifier as a line of code. The artifact it produced was this:

```python
__all__ = []

from . import primes_up_to
```

A directory, its module and its exports are facets of one artifact rather
than independent outcomes, and isolating them destroys the coherence that
authoring requires. This is the first mechanism in the project that we had to
limit rather than extend. It also reconciles with the NL2Repo-Bench finding
that a persistent task tracker correlates with success, since the benefit
lies in planning and the harm lies in fragmentation, and the two are
separable.

Greenfield results after disabling decomposition on an empty workspace, along
with the fixes in section 6.4:

| configuration | success | tool reliability | wasted cycle | tool calls |
| --- | --- | --- | --- | --- |
| baseline | 0/2 | 50 percent | 34 percent | 44 |
| grounded errors, no split | 0/4 | 71 percent | 6 percent | 17 |
| plus forward reference | 1/4 | 86 percent | 8 percent | 16 |

### 6.4 Two negative results about verification itself

**Our L2 rung caused the failure that our L4 check detected.** Building a
package, the model produced this sequence:

```
write mathkit/__init__.py ""                          ok
write "from .primes import is_prime, primes_up_to"    rejected
write "from primes import is_prime, primes_up_to"     rejected
write "from . import primes"                          ok
write mathkit/primes.py                               ok
```

The rejected line was correct. The module it imported was one write away. L2
asks whether an import resolves at this instant, and a package is built over
several writes. Refused twice, the model settled for `from . import primes`,
which exports nothing, and so produced exactly the empty package defect that
our new check then reported. We relaxed L2 to permit a forward reference from
a new package's own `__init__.py` to a direct sibling while the package holds
no other modules, and left L4 to catch anything that never materialises.

**A prompt example is an instruction.** We added guidance pointing renames at
the correct tool and illustrated it with a bare arguments object rather than
a complete call. The model reproduced that shape as prose, inline call
recovery could not parse it without a name field, and steps ended having done
nothing at all. Tool calls per run fell from three or four to one and tool
reliability fell from 92 to 75 percent, while task success rate did not move,
which is why it went unnoticed.

### 6.5 A predicted negative result, confirmed

Late in the campaign we enabled self briefing, in which the model reasons
about the request before changing anything, for the default effort level. It
is an appealing mechanism and it is what the interface displays as a chain of
thought. We then found the NL2Repo-Bench result quoted in section 2 and
flagged our own change as suspect rather than as an improvement.

Ablating it on the three tasks that had regressed:

| measure | briefing on | briefing off |
| --- | --- | --- |
| t3-wire-validator | 0/3 | 2/3 |
| false verification | 47.6 percent | 14.3 percent |
| act don't tell | 94.3 percent | 100 percent |

Three independent signals move together, which is far stronger evidence than
the single task delta would be on its own. A 7B that reasons about work
before doing it talks itself into having done it. The mechanism does not
merely fail to help. It feeds the honesty failure that is this system's worst
metric. The gate now defaults to off and the measurement sits in a comment
beside the default so that nobody re enables it because it sounds sensible.

One distinction the experiment does not settle is worth keeping. A separate
thinker model briefing the coder is not the same proposition as a model
briefing itself, and that path remains available.

This is the clearest instance of the methodological point in section 5. The
mechanism was plausible, it was already shipped, and only an ablation could
distinguish "this helps" from "this is why three tasks regressed".

### 6.6 A metric that catches what task success rate hides

Empty step rate, the share of focused steps that changed nothing, was added
after a regression that scored 100 percent tool reliability, zero percent
wasted cycles and 100 percent act don't tell while doing half the work it had
been asked for. ESR read 61 percent on that run and 5 percent after the fix.
A behavioural metric suite is not decoration. It is the only thing that
separates a clean run from a run that barely happened.

### 6.7 External calibration on SWE-bench

We ran five SWE-bench Lite instances from the smaller repositories in the
set, generating patches locally and evaluating them inside the official per
instance Docker images on the official criterion.

| instance | resolved | target tests passing | patch |
| --- | --- | --- | --- |
| pallets__flask-4045 | no | 0/2 | 913 bytes |
| pallets__flask-4992 | no | 0/1 | 1750 bytes |
| psf__requests-1963 | no | 0/7 | 860 bytes |
| psf__requests-2674 | no | 0/12 | 1876 bytes |
| psf__requests-3362 | no | 0/1 | 6172 bytes |

**Resolved: 0/5. Produced a patch: 5/5.**

The absolute number is what we predicted and it is not the interesting part.
Three things in this table are.

First, every patch applied. The evaluation reached the test stage on all five
instances, meaning the diffs were well formed against the real repository at
the real base commit. The scaffold does what it was built to do. Nothing
here failed because Forge emitted something unusable.

Second, no target test passed anywhere. This is not a near miss that better
verification would convert. Reading the patches shows why: on flask-4045 the
model appended `import flask` to an unrelated file under `examples/` and then
wrote a new test into `tests/test_blueprints.py`, having been told explicitly
that the tests already exist. The failure is localisation. Asked to fix a
described behaviour in an unfamiliar repository of a few hundred files, a 7B
does not find the responsible code, and no amount of checking a change
against the repository helps when the change is in the wrong file.

Third, that is a clean statement of where our thesis stops. Every mechanism
in section 3 answers the question "is this change sound relative to the
repository". None of them answers "is this the right place to change". On
SWE-micro the target file is usually implied by the request, so the question
never arises; on SWE-bench it is most of the problem. We think this is the
honest boundary of scaffolding as capability, and it is a more useful result
than a slightly better score would have been.

It also sharpens the crossover experiment in section 9. The prediction it
makes is specific: as model size rises, localisation stops being the binding
constraint, and only then do the section 3 mechanisms start to show on
SWE-bench. If that is what the matrix shows, the conditional claim in section
2 has a measured shape rather than a rhetorical one.

One adapter defect worth recording, because the score would otherwise have
absorbed it silently. Our first version passed the model's whole diff to the
evaluator, including its edits to test files, which collide with the held out
test patch. Every SWE-bench harness discards test changes and ours did not.
We now strip them, and we note here what stripping hides: on the first
instance we ever ran, the model was told the tests already existed and wrote
one anyway.

---

## 7. Findings we believe generalise

1. An instruction to the model is a hope. A filter on its output is a
   guarantee. Three prompt level rules in this campaign lost to mechanical
   ones covering the same case, including one written an hour earlier.
2. Never gate an inherently multi step operation at each step. We paid for
   this three times: renames, package construction, and per write dangling
   reference checks, the last of which cost 3/6 falling to 1/6.
3. A per step check has to be computed from the step and never from
   cumulative state that earlier steps have already written into. A skipped
   rename looked satisfied because a previous step had touched the same file.
4. The first message of a step is not a verdict. A 7B asked to do one thing
   will often narrate before acting, and a loop that treats narration as
   termination discards steps that were about to succeed. In one run every
   first attempt was discarded this way.
5. When a mechanism appears not to work, check whether it ran at all before
   concluding anything about the model. Three of the last five defects were
   mechanisms that were never reached.
6. Validate a new check against real code before trusting it. Ours run over
   this repository's own source and its commit history, because a check that
   fires on correct code trains the model to ignore the channel.
7. Ablate a plausible mechanism before believing it. Section 6.5 is the case
   in point, and it was already shipped.

---

## 8. Threats to validity

**Scale.** Ten tasks, three seeds, one model, one language. Individual per
task deltas at n=3 are not significant. The aggregate and the mechanism level
traces carry the argument, not any single cell.

**Benchmark authorship.** We wrote the benchmark we are measured on. The
mitigations are hidden checks, verified reference solutions, and a test
asserting that every task fails on the untouched fixture. It remains a real
threat, which is why the calibration in section 6.7 matters more than its
absolute number will suggest.

**Stale ablations.** The gate matrix has not been run again since most of the
mechanisms in section 3 landed. We make no causal claim for any individual
gate beyond section 6.5, where the ablation was actually run.

**False verification.** At 47.6 percent this is the honest headline weakness.
Forge claimed verification it had not performed in nearly half the turns that
claimed anything, including one reply that fabricated a terminal transcript.
The fix, which is to run the checks when the model claims they ran, is
implemented and brought FVR to 14.3 percent in the section 6.5 ablation, but
has not been measured across the full suite.

---

## 9. Future work

1. **The crossover experiment.** The same ablation matrix at several model
   sizes, locating the point for each gate where scaffolding stops paying.
   Forge already speaks to any OpenAI compatible endpoint, so this costs
   inference credits rather than hardware. This settles the disagreement in
   section 2 and is the strongest single experiment available to us.
2. **Interface contracts for greenfield work.** A machine checkable manifest
   of files, the public names each must define, and who imports whom,
   verified against the built tree. This targets the failure mode that
   NL2Repo-Bench identifies as dominant and that our own traces confirm.
3. **Fold step summaries back** into the main trajectory instead of
   discarding them, recovering coherence at the token cost Context-Folding
   reports.
4. **Full SWE-bench Lite** rather than a probe, to bound the benchmark
   authorship threat properly.

---

## Appendix A: reproduction

```bash
forge eval --seeds 3                       # full suite, all tiers
forge eval --tier 5 --seeds 2              # greenfield only
forge eval --ablation no-intent-brief      # the section 6.5 result
forge sweep                                # ablation matrix
forge swebench --limit 5                   # external calibration
```

Traces are written per seed under
`.forge/evals/seed-<n>/<task>/.forge/logs/`, and every result in section 6
can be reconstructed from them.
