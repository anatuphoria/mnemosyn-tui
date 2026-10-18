import json

import pytest

from atlas_package import tiny_structures, write_package
from cns_atlas import Atlas, AtlasError
from cns_atlas.knowledge import Reference
from cns_atlas.timelines import Curve, Mark, Timeline, note_parts

TOY = Timeline("mood", 60.0, "min", 6.0, "draft", "Toy curves [ref2020].",
               (Curve("Toy hormone", (235, 200, 90), ((0.0, 0.1), (20.0, 1.0), (60.0, 0.2))),),
               (Mark(0.0, 1), Mark(20.0, 3), Mark(40.0, 4)))


def test_round_trip():
    assert Timeline.from_json(TOY.to_json()) == TOY


@pytest.mark.parametrize(("time", "step"), [(0, 0), (10, 0), (20, 2), (39.9, 2), (40, 3), (60, 3)])
def test_step_at_follows_the_last_mark(time, step):
    assert TOY.step_at(time) == step


def test_step_before_the_first_mark_is_its_step():
    late = Timeline("x", 10.0, "h", 5.0, "draft", "", TOY.curves, (Mark(5.0, 2),))
    assert late.step_at(1.0) == 1


def test_time_of_a_step():
    assert TOY.time_of(2) == 20.0 and TOY.time_of(1) is None


def test_note_parts_number_citations():
    refs = {"ref2020": Reference("ref2020", "Author A", "A reference", 2020, url="https://x.org")}
    text, found = note_parts("Shapes [ref2020] and [missing].", refs)
    assert text == "Shapes [1] and [missing]." and [r.key for r in found] == ["ref2020"]


def test_atlas_reads_timelines_and_old_packages_have_none(tmp_path):
    root = write_package(tmp_path / "pkg", tiny_structures())
    assert Atlas.open(root).timelines == {}
    (root / "timelines.json").write_text(json.dumps([TOY.to_json()]))
    assert Atlas.open(root).timelines == {"mood": TOY}


def test_malformed_timelines_fail_to_open(tmp_path):
    root = write_package(tmp_path / "pkg", tiny_structures())
    (root / "timelines.json").write_text(json.dumps([{"route": "mood"}]))
    with pytest.raises(AtlasError):
        Atlas.open(root)
