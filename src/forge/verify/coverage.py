"""Requirement coverage — the rung that checks the REQUEST, not the action.

Every other gate asks whether what the agent did was sound: does it parse, do
its names resolve, did it edit real text, did it truly run the tests. None of
them asks the question that actually failed in the benchmark: *was the whole
request covered?*

The observed failure: given "add validate_email, then use it in register",
the model adds the helper, drifts onto self-invented work, and stops. The
code it wrote is valid, resolves, and landed on real text — every gate is
green, and half the request is missing.

Raising effort to `genius` did not fix it (0/4, identical to the default),
which is the point: "re-read the request" is an instruction, and instructions
are what small models drop. So coverage is assessed the same way every other
verdict in Forge is — against evidence. The request is split into atomic
requirements up front, and each is judged against the REAL diff and the
commands that actually ran, never against the model's recollection.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from forge.agents.structured import StructuredOutputError, structured_call
from forge.llm.base import LLMClient, Usage

_MAX_REQUIREMENTS = 6
_MAX_DIFF_CHARS = 4_000

_DECOMPOSE_SYSTEM = """You split a software request into the separate things \
that must be TRUE when it is finished.

Rules:
- One requirement per distinct outcome. "Add X and use it in Y" is TWO.
- Each must be checkable by looking at code: name the function, file or
  behaviour involved.
- Only what the user asked for. Never invent extra work such as writing
  tests, adding documentation or refactoring, unless the user asked.
- At most 6. Most requests have one or two.

Respond with ONLY this JSON:
{"requirements": [{"id": 1, "text": "<what must be true>"}]}"""

_ASSESS_SYSTEM = """You check whether each requirement is satisfied by the \
work that was ACTUALLY done.

You are given the requirements and the real evidence: the unified diff of
every change and the commands that ran. Judge ONLY from that evidence.

- met = true only when the evidence shows it is done. If the diff does not
  contain the change, it is NOT met, no matter what the summary claims.
- If the evidence is empty, nothing is met.
- reason: one short sentence citing the evidence.

Respond with ONLY this JSON:
{"items": [{"id": 1, "met": true, "reason": "<why>"}]}"""


# Decomposing costs an LLM call, and "fix the typo in app.py" cannot be
# partially covered. This cheap filter runs first so only requests that
# plausibly contain several requirements pay for the check.
_MULTI_SIGNALS = re.compile(
    r"\b(?:and then|then|also|as well as|additionally|plus|afterwards)\b"
    r"|\band\b(?=[^.]*\b(?:add|use|call|update|make|fix|rename|remove|create|"
    r"raise|return|handle|check|move|wire|set)\b)"
    r"|\b(?:both|each|every one of)\b"
    r"|;|\bfirst\b.*\bsecond\b|\b\d\.\s",
    re.IGNORECASE,
)


def looks_multi_requirement(request: str) -> bool:
    """True when the request plausibly asks for more than one outcome."""
    return _MULTI_SIGNALS.search(request) is not None


class Requirement(BaseModel):
    id: int
    text: str


class RequirementList(BaseModel):
    requirements: list[Requirement] = Field(default_factory=list)


class CoverageItem(BaseModel):
    id: int
    met: bool
    reason: str = ""


class CoverageVerdict(BaseModel):
    items: list[CoverageItem] = Field(default_factory=list)

    def unmet(self, requirements: list[Requirement]) -> list[Requirement]:
        missed = {item.id for item in self.items if not item.met}
        return [r for r in requirements if r.id in missed]


def decompose(llm: LLMClient, request: str, usage: Usage | None = None) -> list[Requirement]:
    """Split a request into atomic, checkable requirements. Returns [] when
    the model cannot produce a usable list — coverage then simply does not
    apply, which is safer than blocking on a broken decomposition."""
    try:
        result = structured_call(
            llm, _DECOMPOSE_SYSTEM, f"Request:\n{request}", RequirementList, usage=usage
        )
    except (StructuredOutputError, Exception):  # noqa: BLE001 — never break the turn
        return []
    return result.requirements[:_MAX_REQUIREMENTS]


def build_evidence(diff: str, changed_files: list[str], commands: list[str]) -> str:
    """The ground truth a coverage judgement is allowed to use."""
    if not diff and not changed_files:
        return "No files were changed and no commands were run."
    sections = []
    if changed_files:
        sections.append("Files changed: " + ", ".join(changed_files))
    if commands:
        sections.append("Commands run:\n" + "\n".join(f"$ {c}" for c in commands))
    if diff:
        clipped = diff[:_MAX_DIFF_CHARS]
        if len(diff) > _MAX_DIFF_CHARS:
            clipped += "\n... [diff truncated]"
        sections.append("Unified diff of every change:\n" + clipped)
    return "\n\n".join(sections)


def assess(
    llm: LLMClient,
    requirements: list[Requirement],
    evidence: str,
    usage: Usage | None = None,
) -> CoverageVerdict:
    """Judge each requirement against the evidence. On failure returns an
    empty verdict, which is treated as 'cannot tell' rather than 'unmet'."""
    if not requirements:
        return CoverageVerdict()
    listing = "\n".join(f"{r.id}. {r.text}" for r in requirements)
    message = f"## Requirements\n{listing}\n\n## Evidence\n{evidence}"
    try:
        return structured_call(llm, _ASSESS_SYSTEM, message, CoverageVerdict, usage=usage)
    except (StructuredOutputError, Exception):  # noqa: BLE001
        return CoverageVerdict()


def coverage_nudge(unmet: list[Requirement]) -> str:
    listing = "\n".join(f"- {r.text}" for r in unmet)
    return (
        "Your work does not yet cover the whole request. These parts are NOT "
        f"done — the diff does not contain them:\n{listing}\n\n"
        "Do them NOW with tools, in the files that already exist. Do not add "
        "anything the user did not ask for, and do not repeat work that is "
        "already done."
    )
