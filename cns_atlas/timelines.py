from __future__ import annotations

import bisect
from dataclasses import dataclass

from .knowledge import CITATION, Reference


@dataclass(frozen=True)
class Curve:
    name: str
    color: tuple[int, int, int]
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Mark:
    time: float
    step: int


@dataclass(frozen=True)
class Timeline:
    route: str
    duration: float
    unit: str
    play_seconds: float
    status: str
    note: str
    curves: tuple[Curve, ...]
    marks: tuple[Mark, ...]

    def step_at(self, time: float) -> int:
        index = bisect.bisect_right([mark.time for mark in self.marks], time) - 1
        return self.marks[max(index, 0)].step - 1

    def time_of(self, step: int) -> float | None:
        return next((mark.time for mark in self.marks if mark.step - 1 == step), None)

    def to_json(self) -> dict:
        return {"route": self.route, "duration": self.duration, "unit": self.unit,
                "play_seconds": self.play_seconds, "status": self.status, "note": self.note,
                "curves": [{"name": c.name, "color": list(c.color),
                            "points": [list(p) for p in c.points]} for c in self.curves],
                "marks": [{"time": m.time, "step": m.step} for m in self.marks]}

    @classmethod
    def from_json(cls, data: dict) -> Timeline:
        curves = tuple(Curve(str(c["name"]), tuple(int(v) for v in c["color"]),
                             tuple((float(t), float(v)) for t, v in c["points"]))
                       for c in data["curves"])
        marks = tuple(Mark(float(m["time"]), int(m["step"])) for m in data["marks"])
        return cls(str(data["route"]), float(data["duration"]), str(data["unit"]),
                   float(data["play_seconds"]), str(data["status"]), str(data["note"]),
                   curves, marks)


def note_parts(note: str, references: dict[str, Reference]) -> tuple[str, tuple[Reference, ...]]:
    keys = [key for key in dict.fromkeys(CITATION.findall(note)) if key in references]
    order = {key: n for n, key in enumerate(keys, 1)}
    text = CITATION.sub(lambda m: f"[{order[m.group(1)]}]" if m.group(1) in order else m.group(0),
                        note)
    return text, tuple(references[key] for key in keys)
