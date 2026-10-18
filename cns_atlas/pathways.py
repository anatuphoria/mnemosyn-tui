from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .knowledge import Entry, Reference
from .model import Pathway, Station, Step, Structure

START_SIDES = ("left", "right")
STYLE_LABELS = {"lesion": (("Role:", "Lesion:"), ()),
                "behaviour": (("Role:", "Disorders:"), ("Drugs:",))}
ALL_LABELS = ("Role:", "Lesion:", "Disorders:", "Drugs:")
SIDE_WORD = re.compile(r"\{([^{}\n]*)\}")


@dataclass(frozen=True)
class StepText:
    role: str
    lesion: str = ""
    disorders: str = ""
    drugs: str = ""


@dataclass(frozen=True)
class PathwayText:
    status: str
    overview: str
    steps: tuple[StepText, ...]
    references: tuple[Reference, ...]


def other_side(side: str) -> str:
    return {"left": "right", "right": "left"}[side]


def side_of(step_side: str, start_side: str) -> str:
    if step_side == "midline":
        return "midline"
    return start_side if step_side == "same" else other_side(start_side)


def lit(structures: Sequence[Structure], has_shape: Callable[[int], bool], step: Step,
        start_side: str) -> tuple[int, ...]:
    if not isinstance(step, Station):
        return ()
    sides = {side_of(step.side, start_side), "midline", "none"}
    wanted = set(step.structures)
    return tuple(s.id for s in structures
                 if s.side in sides and wanted.intersection(s.structure_id_path)
                 and has_shape(s.id))


def resolve(atlas, step: Step, start_side: str) -> tuple[int, ...]:
    return lit(atlas.structures, atlas.has_mesh, step, start_side)


def lit_both(structures: Sequence[Structure], has_shape: Callable[[int], bool],
             step: Step) -> tuple[int, ...]:
    found = {*lit(structures, has_shape, step, "left"), *lit(structures, has_shape, step, "right")}
    return tuple(s.id for s in structures if s.id in found)


def resolve_both(atlas, step: Step) -> tuple[int, ...]:
    return lit_both(atlas.structures, atlas.has_mesh, step)


def lights(atlas, pathway: Pathway, side: str) -> list[tuple[int, ...]]:
    if pathway.sides == "both":
        return [resolve_both(atlas, step) for step in pathway.steps]
    return [resolve(atlas, step, side) for step in pathway.steps]


def side_words(text: str) -> list[str]:
    return SIDE_WORD.findall(text)


def fill(text: str, side: str) -> str:
    return text.replace("{same}", side).replace("{opposite}", other_side(side))


def unknown_side_words(text: str) -> list[str]:
    return [word for word in SIDE_WORD.findall(text) if word not in ("same", "opposite")]


def step_problems(body: str, style: str = "lesion") -> list[str]:
    required, optional = STYLE_LABELS[style]
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    problems = []
    for label in (*required, *optional):
        found = [line for line in lines if line.startswith(label)]
        if label in required and len(found) != 1:
            problems.append(f"needs one '{label}' line, found {len(found)}")
        elif len(found) > 1:
            problems.append(f"needs at most one '{label}' line, found {len(found)}")
        elif found and not found[0][len(label):].strip():
            problems.append(f"'{label}' line is empty")
    problems += [f"unexpected line {line!r}" for line in lines
                 if not line.startswith((*required, *optional))]
    return problems


def step_text(body: str) -> StepText:
    lines = [line.strip() for line in body.splitlines()]
    return StepText(*(next((line[len(label):].strip() for line in lines if line.startswith(label)),
                           "") for label in ALL_LABELS))


def text_for(entry: Entry, count: int, side: str, sides: str = "one") -> PathwayText:
    if side not in START_SIDES:
        raise ValueError(f"side must be left or right, got {side!r}")

    def show(text: str) -> str:
        return entry.numbered(fill(text, side) if sides == "one" else text)

    parsed = (step_text(entry.sections.get(f"Step {n}", "")) for n in range(1, count + 1))
    steps = tuple(StepText(show(t.role), show(t.lesion), show(t.disorders), show(t.drugs))
                  for t in parsed)
    return PathwayText(entry.status, show(entry.sections.get("Overview", "")), steps,
                       entry.references)
