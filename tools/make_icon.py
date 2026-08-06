"""Generate the Forge application icon.

The mark is the same four point spark the interface uses, so the desktop
icon, the window and the wordmark are one thing rather than three.

Written with the standard library only. Pillow would be one import and a
runtime dependency for something that is generated once and committed, which
is a poor trade. PNG is deflate plus a few length prefixed chunks, and an ICO
is a small directory of PNGs, so both are short.

    python tools/make_icon.py

writes src/forge/server/static/forge.ico.
"""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

ACCENT = (217, 119, 87)  # the terracotta used everywhere in the interface
SIZES = (16, 24, 32, 48, 64, 128, 256)
_SUPERSAMPLE = 4  # draw large and average down, so the points stay smooth


def spark(size: float, exponent: float = 3.0, steps: int = 720) -> list[tuple[float, float]]:
    """The concave four point star, as a polygon.

    An astroid is exactly this shape: points on the axes and sides that curve
    inward. Raising cosine and sine to a power above one pulls the sides in,
    and three matches the mark drawn in the interface.
    """
    radius = size / 2
    points: list[tuple[float, float]] = []
    for step in range(steps):
        angle = 2 * math.pi * step / steps
        cos_t, sin_t = math.cos(angle), math.sin(angle)
        x = math.copysign(abs(cos_t) ** exponent, cos_t)
        y = math.copysign(abs(sin_t) ** exponent, sin_t)
        points.append((radius + x * radius, radius - y * radius))
    return points


def _coverage(size: int) -> list[list[float]]:
    """How much of each pixel the spark covers, via a supersampled scanline."""
    big = size * _SUPERSAMPLE
    polygon = spark(big)
    mask = [[0.0] * size for _ in range(size)]
    edges = [
        (polygon[i], polygon[(i + 1) % len(polygon)])
        for i in range(len(polygon))
    ]
    for row in range(big):
        y = row + 0.5
        crossings: list[float] = []
        for (x0, y0), (x1, y1) in edges:
            if (y0 <= y < y1) or (y1 <= y < y0):
                crossings.append(x0 + (y - y0) * (x1 - x0) / (y1 - y0))
        crossings.sort()
        for start, end in zip(crossings[0::2], crossings[1::2], strict=False):
            for column in range(max(0, int(start)), min(big, int(end) + 1)):
                if start <= column + 0.5 <= end:
                    mask[row // _SUPERSAMPLE][column // _SUPERSAMPLE] += 1.0
    scale = _SUPERSAMPLE * _SUPERSAMPLE
    return [[min(1.0, value / scale) for value in line] for line in mask]


def _png(size: int) -> bytes:
    """A 32 bit RGBA PNG of the spark at this size."""
    mask = _coverage(size)
    raw = bytearray()
    for row in mask:
        raw.append(0)  # filter type 0 for the scanline
        for value in row:
            alpha = int(round(value * 255))
            raw += bytes((*ACCENT, alpha))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def build(destination: Path) -> Path:
    """Write a multi resolution .ico. Windows picks the size it needs."""
    images = [(size, _png(size)) for size in SIZES]
    offset = 6 + 16 * len(images)
    directory = bytearray(struct.pack("<HHH", 0, 1, len(images)))
    body = bytearray()
    for size, data in images:
        directory += struct.pack(
            "<BBBBHHII",
            0 if size >= 256 else size,  # 0 means 256 in the ICO format
            0 if size >= 256 else size,
            0,
            0,
            1,
            32,
            len(data),
            offset,
        )
        body += data
        offset += len(data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(bytes(directory) + bytes(body))
    return destination


if __name__ == "__main__":
    target = Path(__file__).resolve().parent.parent / "src/forge/server/static/forge.ico"
    written = build(target)
    print(f"wrote {written} ({written.stat().st_size} bytes, sizes {SIZES})")
