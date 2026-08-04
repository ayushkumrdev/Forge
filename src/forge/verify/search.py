"""Execution-guided candidate search — test-time compute for a fixed model.

The benchmark left a clear boundary: mechanical failure is essentially gone
(tool reliability 90%, wasted cycles 2%, no hallucinated identifiers) and the
agent still solves the wrong problem sometimes. Gates cannot fix that; they
verify an action, they cannot supply a better idea.

What can, without a larger model, is trying more than once. A 7B sampled
twice at different temperatures produces genuinely different attempts, and
Forge can already *judge* an attempt against reality — the verification
ladder says whether it is sound, and requirement coverage says whether it
did what was asked. So: run k attempts in isolation, score each against
those two, keep the best, discard the rest.

Isolation is the hard part and is deliberately simple here: snapshot the
files, let a candidate run against the real workspace, record what it did,
then restore. No overlay filesystem, no path rewriting — the ledger already
proves that restoring bytes exactly is reliable, and a candidate that
crashes leaves nothing behind.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# Directories never worth snapshotting: build output, dependencies, VCS and
# Forge's own state. A snapshot that walks node_modules would cost more than
# the search saves.
_SKIP_DIRS = frozenset(
    {
        ".git", ".forge", ".venv", "venv", "node_modules", "__pycache__",
        ".pytest_cache", ".ruff_cache", ".mypy_cache", "dist", "build",
        ".idea", ".vscode", ".tox", "target",
    }
)
_MAX_SNAPSHOT_FILES = 400
_MAX_SNAPSHOT_BYTES = 8 * 1024 * 1024


@dataclass
class Snapshot:
    """Byte-exact state of a workspace, restorable after a candidate runs."""

    root: Path
    files: dict[Path, bytes] = field(default_factory=dict)
    complete: bool = True  # False when the tree was too large to capture

    def restore(self) -> None:
        """Put the workspace back exactly as it was, including deleting any
        file the candidate created."""
        for path, data in self.files.items():
            try:
                if path.read_bytes() != data:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
            except OSError:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                except OSError:
                    pass
        for path in _walk(self.root):
            if path not in self.files:
                with contextlib.suppress(OSError):
                    path.unlink()

    def changed_since(self) -> list[Path]:
        changed = []
        for path in _walk(self.root):
            before = self.files.get(path)
            try:
                now = path.read_bytes()
            except OSError:
                continue
            if before != now:
                changed.append(path)
        return changed


def _walk(root: Path) -> list[Path]:
    out: list[Path] = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.name in _SKIP_DIRS:
                continue
            if entry.is_dir():
                stack.append(entry)
            elif entry.is_file():
                out.append(entry)
    return out


def capture(root: Path) -> Snapshot:
    """Snapshot a workspace, or report incompleteness rather than lie about
    being able to restore it."""
    snapshot = Snapshot(root=root)
    total = 0
    for path in _walk(root):
        if len(snapshot.files) >= _MAX_SNAPSHOT_FILES or total > _MAX_SNAPSHOT_BYTES:
            snapshot.complete = False
            break
        try:
            data = path.read_bytes()
        except OSError:
            continue
        total += len(data)
        snapshot.files[path] = data
    return snapshot


def _search_temperatures(base: float, count: int) -> list[float | None]:
    """First attempt at the configured temperature, then progressively hotter
    so the samples actually differ — two identical attempts buy nothing."""
    temps: list[float | None] = [None]
    for step in range(1, max(count, 1)):
        temps.append(round(min(base + 0.3 * step, 0.9), 2))
    return temps


@dataclass
class Candidate:
    """One attempt: what it changed and how well it scored."""

    index: int
    temperature: float | None
    changed: list[str] = field(default_factory=list)
    verified: bool = False  # every write it made passed the ladder
    satisfied: bool = False  # requirement coverage says it did the job
    files: dict[Path, bytes] = field(default_factory=dict)  # its resulting state
    error: str | None = None

    @property
    def score(self) -> tuple[int, int, int]:
        """Higher is better. Doing the job dominates; among attempts that did
        it, prefer the one that verified, then the smaller change — a diff
        that touches less is easier to review and less likely to break
        something the checks do not cover."""
        return (
            int(self.satisfied),
            int(self.verified),
            -len(self.changed),
        )

    def apply(self) -> None:
        for path, data in self.files.items():
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            except OSError:
                pass


def search(
    workspace: Path,
    attempt: Callable[[int, float | None], tuple[bool, bool]],
    temperatures: list[float | None],
    on_candidate: Callable[[Candidate], None] | None = None,
) -> Candidate | None:
    """Run one attempt per temperature in isolation and return the best.

    `attempt(index, temperature)` performs the work against the real
    workspace and reports (changed_anything, satisfied). The workspace is
    restored between attempts and the winner is re-applied at the end, so
    only the best attempt survives.

    Returns None when the workspace is too large to snapshot safely — the
    caller then simply runs once, which is the behaviour without search."""
    base = capture(workspace)
    if not base.complete:
        return None

    candidates: list[Candidate] = []
    for index, temperature in enumerate(temperatures):
        candidate = Candidate(index=index, temperature=temperature)
        try:
            changed_anything, satisfied = attempt(index, temperature)
            candidate.satisfied = satisfied
            candidate.verified = changed_anything
        except Exception as exc:  # noqa: BLE001 — a bad candidate is not a crash
            candidate.error = f"{type(exc).__name__}: {exc}"
        candidate.changed = [
            str(p.relative_to(workspace)).replace("\\", "/")
            for p in base.changed_since()
        ]
        candidate.files = {p: p.read_bytes() for p in _walk(workspace)}
        candidates.append(candidate)
        if on_candidate is not None:
            on_candidate(candidate)
        base.restore()
        # a candidate that already did the job cleanly is not worth beating
        if candidate.satisfied and candidate.verified:
            break

    winner = max(
        (c for c in candidates if c.changed and c.error is None),
        key=lambda c: c.score,
        default=None,
    )
    if winner is not None:
        winner.apply()
    return winner
