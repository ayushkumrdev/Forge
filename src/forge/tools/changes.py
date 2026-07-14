"""Change ledger: records the original content of every file before Forge
modifies it, writes a physical backup under .forge/backups/<run_id>/, and can
produce a unified diff of everything that changed. This is what guarantees
"never overwrite user work without backups"."""

from __future__ import annotations

import difflib
from pathlib import Path


class ChangeLedger:
    def __init__(self, workspace: Path, run_id: str) -> None:
        self.workspace = workspace.resolve()
        self.backup_dir = self.workspace / ".forge" / "backups" / run_id
        # None means the file did not exist before this run.
        self._originals: dict[Path, str | None] = {}

    def record_before_write(self, path: Path) -> None:
        """Call before every write/edit. Idempotent per file per run."""
        path = path.resolve()
        if path in self._originals:
            return
        if path.exists():
            original = path.read_text(encoding="utf-8-sig", errors="replace")
            self._originals[path] = original
            backup_path = self.backup_dir / path.relative_to(self.workspace)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            backup_path.write_text(original, encoding="utf-8")
        else:
            self._originals[path] = None

    @property
    def changed_files(self) -> list[str]:
        return sorted(
            str(path.relative_to(self.workspace)).replace("\\", "/")
            for path in self._originals
        )

    def restore_all(self) -> list[str]:
        """Undo: put every touched file back to its pre-run state. Files that
        did not exist before are deleted. Returns the restored paths."""
        restored: list[str] = []
        for path, original in sorted(self._originals.items()):
            if original is None:
                path.unlink(missing_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(original, encoding="utf-8")
            restored.append(str(path.relative_to(self.workspace)).replace("\\", "/"))
        self._originals.clear()
        return restored

    def unified_diff(self) -> str:
        """Diff of all recorded files: original content vs. current on-disk."""
        chunks: list[str] = []
        for path, original in sorted(self._originals.items()):
            relative = str(path.relative_to(self.workspace)).replace("\\", "/")
            before = (original or "").splitlines(keepends=True)
            current = (
                path.read_text(encoding="utf-8-sig", errors="replace").splitlines(keepends=True)
                if path.exists()
                else []
            )
            diff = difflib.unified_diff(
                before,
                current,
                fromfile=f"a/{relative}" if original is not None else "/dev/null",
                tofile=f"b/{relative}" if path.exists() else "/dev/null",
            )
            chunks.append("".join(diff))
        return "\n".join(chunk for chunk in chunks if chunk)
