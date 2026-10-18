from __future__ import annotations

import numpy as np
from rich.color import Color
from rich.segment import Segment
from rich.style import Style
from textual.strip import Strip

HALF_BLOCK = "▀"
_MAX_CACHED_STYLES = 4096


def pack_cells(rgb: np.ndarray) -> np.ndarray:
    """(2*rows, cols, 3) pixels -> (rows, cols) int64 (fg, bg) keys."""
    if rgb.shape[0] % 2:
        rgb = rgb[:-1]
    px = rgb.astype(np.int64)
    top, bot = px[0::2], px[1::2]
    return ((top[..., 0] << 40) | (top[..., 1] << 32) | (top[..., 2] << 24)
            | (bot[..., 0] << 16) | (bot[..., 1] << 8) | bot[..., 2])


class HalfBlockEncoder:

    def __init__(self) -> None:
        self._styles: dict[int, Style] = {}
        self._prev: np.ndarray | None = None

    def _style(self, key: int) -> Style:
        style = self._styles.get(key)
        if style is None:
            if len(self._styles) >= _MAX_CACHED_STYLES:
                self._styles.clear()
            fg = Color.from_rgb((key >> 40) & 255, (key >> 32) & 255, (key >> 24) & 255)
            bg = Color.from_rgb((key >> 16) & 255, (key >> 8) & 255, key & 255)
            style = self._styles[key] = Style(color=fg, bgcolor=bg)
        return style

    def _encode_row(self, row: np.ndarray) -> Strip:
        breaks = np.flatnonzero(np.diff(row)) + 1
        starts = [0, *breaks.tolist()]
        ends = [*breaks.tolist(), len(row)]
        segments = [Segment(HALF_BLOCK * (end - start), self._style(int(row[start])))
                    for start, end in zip(starts, ends)]
        return Strip(segments, len(row))

    def encode(self, rgb: np.ndarray) -> tuple[list[Strip | None], list[int]]:
        cells = pack_cells(rgb)
        prev = self._prev
        if prev is None or prev.shape != cells.shape:
            changed = list(range(cells.shape[0]))
        else:
            changed = np.flatnonzero((cells != prev).any(axis=1)).tolist()
        self._prev = cells
        strips: list[Strip | None] = [None] * cells.shape[0]
        for y in changed:
            strips[y] = self._encode_row(cells[y])
        return strips, changed
