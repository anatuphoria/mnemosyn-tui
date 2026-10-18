from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image
from textual.strip import Strip

from .present import HalfBlockEncoder
from .renderer import project_centroids, render_scene_dispatch
from .scene import SceneState

MAX_LABELS = 8


@dataclass(frozen=True)
class RenderRequest:
    scene: SceneState
    cols: int
    rows: int
    width: int
    height: int
    hifi: bool
    budget: int
    halfblock: bool


@dataclass
class Frame:
    cols: int
    rows: int
    idbuf: np.ndarray
    region_ids: list[str]
    strips: list[Strip | None]
    changed_rows: list[int]
    image: Image.Image | None
    labels: str
    render_ms: float
    encode_ms: float

    @property
    def total_ms(self) -> float:
        return self.render_ms + self.encode_ms


def label_line(scene: SceneState, width: int, height: int) -> str:
    if not scene.show_labels:
        return ""
    items = []
    for rid, (sx, sy, _) in project_centroids(scene, width, height).items():
        if 0 <= sx <= width and 0 <= sy <= height:
            star = "★ " if rid == scene.selected_id else ""
            dist = abs(sx - width / 2) + abs(sy - height / 2)
            items.append((dist, f"{star}{scene.regions[rid].name}"))
    items.sort(key=lambda t: t[0])
    return " · ".join(name for _, name in items[:MAX_LABELS])


def render_frame(req: RenderRequest, encoder: HalfBlockEncoder) -> Frame:
    t0 = time.perf_counter()
    img, idbuf, ids = render_scene_dispatch(req.scene, req.width, req.height,
                                            hifi=req.hifi, max_pts=req.budget)
    labels = label_line(req.scene, req.width, req.height)
    t1 = time.perf_counter()
    strips: list[Strip | None] = []
    changed: list[int] = []
    if req.halfblock:
        strips, changed = encoder.encode(np.asarray(img))
    t2 = time.perf_counter()
    return Frame(cols=req.cols, rows=req.rows, idbuf=idbuf, region_ids=ids,
                 strips=strips, changed_rows=changed,
                 image=None if req.halfblock else img, labels=labels,
                 render_ms=(t1 - t0) * 1000, encode_ms=(t2 - t1) * 1000)


class RenderWorker:

    def __init__(self, deliver: Callable[[Frame], None],
                 fail: Callable[[BaseException], None]) -> None:
        self._deliver = deliver
        self._fail = fail
        self._encoder = HalfBlockEncoder()
        self._wakeup = threading.Condition()
        self._request: RenderRequest | None = None
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name="brain-render", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, request: RenderRequest) -> None:
        with self._wakeup:
            self._request = request
            self._wakeup.notify()

    def stop(self) -> None:
        with self._wakeup:
            self._stopped = True
            self._wakeup.notify()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._wakeup:
                while self._request is None and not self._stopped:
                    self._wakeup.wait()
                if self._stopped:
                    return
                request, self._request = self._request, None
            try:
                frame = render_frame(request, self._encoder)
            except Exception as exc:
                self._encoder = HalfBlockEncoder()
                self._fail(exc)
                continue
            self._deliver(frame)
