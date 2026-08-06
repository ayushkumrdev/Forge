"""Running child processes without flashing a console window.

On Windows, a subprocess started from a GUI process gets its own console
window unless it is told not to. The desktop app runs the model's shell
commands, git, the test suite and the import checks, so a busy turn threw a
stack of black windows onto the user's screen and stole focus from whatever
they were doing.

`CREATE_NO_WINDOW` suppresses that. It only exists on Windows, so everywhere
else this is a plain passthrough and the flag is never mentioned.

Every process Forge starts goes through here. A single missed call site is a
window in someone's face, which is exactly the kind of thing that gets fixed
in nine places and forgotten in the tenth.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

# 0 on any platform that does not have the flag, so it is safe to OR in
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def hidden_flags(existing: int = 0) -> int:
    """Creation flags with the console window suppressed."""
    return existing | _NO_WINDOW


def run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
    """subprocess.run that never opens a console window.

    Callers that pass their own creationflags keep them; the no-window bit is
    added rather than substituted.
    """
    if _NO_WINDOW:
        kwargs["creationflags"] = hidden_flags(kwargs.get("creationflags", 0))
    return subprocess.run(*args, **kwargs)  # noqa: S603 — callers pass real argv


def popen(*args: Any, **kwargs: Any) -> subprocess.Popen:
    """subprocess.Popen that never opens a console window."""
    if _NO_WINDOW:
        kwargs["creationflags"] = hidden_flags(kwargs.get("creationflags", 0))
    return subprocess.Popen(*args, **kwargs)  # noqa: S603


__all__ = ["hidden_flags", "popen", "run"]
