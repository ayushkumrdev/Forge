"""Launching Forge should not throw a console window on the screen, and the
desktop icon should be the mark the app uses everywhere else.

Both were real complaints. Every shell command, git call, test run and import
check the agent makes is a child process, and on Windows a child process
started from a GUI parent gets its own console window unless told otherwise.
A busy turn threw a stack of them up and stole focus from whatever the user
was doing.
"""

import os
import struct
import subprocess
import zlib
from pathlib import Path

from forge import process
from forge.cli import app_icon


def test_the_process_helper_suppresses_the_console(monkeypatch):
    seen = {}

    def fake_run(*args, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(process.subprocess, "run", fake_run)
    process.run(["git", "status"], capture_output=True)

    if os.name == "nt":
        assert seen["creationflags"] & subprocess.CREATE_NO_WINDOW
    else:  # nothing to suppress, and the flag does not exist
        assert "creationflags" not in seen


def test_caller_creation_flags_are_kept_not_replaced():
    if os.name != "nt":
        return
    existing = subprocess.CREATE_NEW_PROCESS_GROUP
    combined = process.hidden_flags(existing)
    assert combined & existing
    assert combined & subprocess.CREATE_NO_WINDOW


def test_nothing_spawns_a_process_around_the_helper():
    """A single missed call site is a window in someone's face."""
    root = Path(__file__).resolve().parent.parent / "src" / "forge"
    offenders = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if path.name != "process.py"
        and any(
            call in path.read_text(encoding="utf-8")
            for call in ("subprocess.run(", "subprocess.Popen(")
        )
    ]
    assert offenders == [], f"these bypass forge.process: {offenders}"


def test_the_icon_ships_and_is_a_real_multi_resolution_ico():
    icon = app_icon()
    assert icon.is_file(), f"missing {icon}"
    data = icon.read_bytes()
    reserved, kind, count = struct.unpack("<HHH", data[:6])
    assert (reserved, kind) == (0, 1)  # a real ICO header
    assert count >= 4, "Windows picks a size per surface, so ship several"

    sizes = []
    for index in range(count):
        entry = struct.unpack("<BBBBHHII", data[6 + 16 * index : 22 + 16 * index])
        width, _, _, _, _, bpp, length, offset = entry
        sizes.append(width or 256)
        assert bpp == 32, "the mark needs an alpha channel"
        assert data[offset : offset + 8] == b"\x89PNG\r\n\x1a\n"
        assert length > 0
    assert 16 in sizes and 32 in sizes, f"missing a common size: {sizes}"


def _alpha_grid(index: int = 2) -> list[list[int]]:
    data = app_icon().read_bytes()
    entry = struct.unpack("<BBBBHHII", data[6 + 16 * index : 22 + 16 * index])
    width, _, _, _, _, _, length, offset = entry
    png = data[offset : offset + length]
    idat, cursor = b"", 8
    while cursor < len(png):
        size = struct.unpack(">I", png[cursor : cursor + 4])[0]
        if png[cursor + 4 : cursor + 8] == b"IDAT":
            idat += png[cursor + 8 : cursor + 8 + size]
        cursor += 12 + size
    raw = zlib.decompress(idat)
    stride = 1 + width * 4
    return [
        [raw[row * stride + 1 + column * 4 + 3] for column in range(width)]
        for row in range(width)
    ]


def test_the_icon_is_the_spark_and_not_a_blank_square():
    """Solid in the middle, empty in the corners, which is what a four point
    star is and what a placeholder square is not."""
    alpha = _alpha_grid()
    middle = len(alpha) // 2
    assert alpha[middle][middle] > 200, "the middle of the spark should be solid"
    corners = [alpha[0][0], alpha[0][-1], alpha[-1][0], alpha[-1][-1]]
    assert max(corners) == 0, "the corners should be transparent"


def test_the_spark_has_four_points_on_the_axes():
    """The arms reach the edges along the axes and the diagonals do not,
    which is the difference between this mark and a circle."""
    alpha = _alpha_grid()
    size = len(alpha)
    middle = size // 2
    on_axis = max(alpha[1][middle], alpha[size - 2][middle])
    on_diagonal = alpha[size // 4][size // 4]
    assert on_axis > on_diagonal, "the points should sit on the axes"
