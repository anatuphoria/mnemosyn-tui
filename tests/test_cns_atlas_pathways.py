import json

import pytest

from atlas_package import tetra, tiny_structures, write_package
from cns_atlas import Atlas, AtlasError, Crossing, Pathway, Relay, Station
from cns_atlas.model import DEFAULT_GROUP
from cns_atlas.pathways import (StepText, fill, lights, lit, resolve, resolve_both, side_words,
                                step_problems, step_text)

REFERENCES = """\
[ref2020]
authors = "Author A"
title = "A reference"
year = 2020
url = "https://example.org/ref"
"""
LOOP = Pathway("loop", "Toy loop", "alpha gyrus", (
    Station((100,)), Crossing("Toy crossing"), Station((100,), "opposite"),
    Relay("Toy relay", "opposite"), Station((110,), "midline")))
LOOP_TEXT = """\
+++
key = "pathway-loop"
+++
## Overview
Starts in the alpha gyrus [ref2020].

## Step 1
Role: Starts in the {same} alpha gyrus [ref2020].
Lesion: Loss on the {opposite} side.

## Step 2
Role: Crosses.
Lesion: Both sides.

## Step 3
Role: Reaches the {opposite} alpha gyrus.
Lesion: Loss on the {same} side.

## Step 4
Role: A relay the atlas lacks.
Lesion: Not described.

## Step 5
Role: Ends in the beta nucleus.
Lesion: Loss on both sides.
"""


@pytest.fixture
def atlas(tmp_path):
    meshes = {101: tetra(), 102: tetra(20.0), 110: tetra(40.0)}
    root = write_package(tmp_path / "pkg", tiny_structures(), meshes=meshes, pathways=[LOOP],
                         entries={"en": {"pathway-loop": LOOP_TEXT}}, references=REFERENCES)
    return Atlas.open(root)


def test_pathways_load_with_their_steps(atlas):
    assert atlas.pathways == (LOOP,)


@pytest.mark.parametrize(("step", "side", "expected"), [
    (Station((100,)), "left", (101,)),
    (Station((100,)), "right", (102,)),
    (Station((100,), "opposite"), "left", (102,)),
    (Station((110,), "midline"), "right", (110,)),
    (Station((100,), "midline"), "left", ()),
    (Station((110,)), "left", (110,)),
    (Station((2,)), "left", (101, 110)),
    (Station((100, 110), "opposite"), "right", (101, 110)),
    (Crossing("x"), "left", ()),
    (Relay("x", "opposite"), "left", ()),
])
def test_resolve_lights_the_shapes_on_the_resolved_side(atlas, step, side, expected):
    assert resolve(atlas, step, side) == expected


def test_structures_without_a_shape_never_light():
    structures = tiny_structures()
    assert lit(structures, {101}.__contains__, Station((100,), "opposite"), "left") == ()
    assert lit(structures, {101}.__contains__, Station((100,)), "left") == (101,)


def test_side_words_are_filled_for_the_chosen_side():
    text = "the {same} eye and the {opposite} field"
    assert fill(text, "left") == "the left eye and the right field"
    assert fill(text, "right") == "the right eye and the left field"


def test_pathway_text_fills_sides_and_numbers_citations(atlas):
    text = atlas.pathway_text("loop", "right")
    assert text.status == "draft"
    assert text.overview == "Starts in the alpha gyrus [1]."
    assert len(text.steps) == 5
    assert text.steps[0] == StepText("Starts in the right alpha gyrus [1].",
                                     "Loss on the left side.")
    assert text.steps[2].role == "Reaches the left alpha gyrus."
    assert [r.key for r in text.references] == ["ref2020"]


def test_pathway_text_is_none_without_a_pathway_or_a_file(atlas, tmp_path):
    assert atlas.pathway_text("missing", "left") is None
    root = write_package(tmp_path / "bare", tiny_structures(), pathways=[LOOP])
    assert Atlas.open(root).pathway_text("loop", "left") is None


def test_pathway_text_needs_left_or_right(atlas):
    with pytest.raises(ValueError):
        atlas.pathway_text("loop", "same")


def test_step_text_and_its_problems():
    assert step_text("Role: Relays.\nLesion: Deafness.\n") == StepText("Relays.", "Deafness.")
    assert step_text("") == StepText("", "")
    assert step_problems("Role: Relays.\n\nLesion: Deafness.") == []
    assert step_problems("Role: Relays.") == ["needs one 'Lesion:' line, found 0"]
    assert step_problems("Role: A.\nRole: B.\nLesion: C.") == ["needs one 'Role:' line, found 2"]
    assert step_problems("Role:\nLesion: C.") == ["'Role:' line is empty"]
    assert step_problems("Role: A.\nLesion: C.\nMore text.") == ["unexpected line 'More text.'"]


@pytest.mark.parametrize("record", [
    {"key": "x", "name": "X", "start": "eye", "steps": [{"structures": [100], "side": "up"}]},
    {"key": "x", "name": "X", "start": "eye", "steps": [{"cross": "A", "relay": "B"}]},
    {"key": "x", "name": "X", "steps": []},
    {"key": "x", "name": "X", "start": "eye", "steps": [5]},
])
def test_malformed_pathway_records_fail_to_open(tmp_path, record):
    root = write_package(tmp_path / "pkg", tiny_structures())
    (root / "pathways.json").write_text(json.dumps([record]))
    with pytest.raises(AtlasError):
        Atlas.open(root)


MOOD = Pathway("mood", "Toy mood", "beta nucleus", (
    Station((110,)), Relay("Toy relay"), Station((100,))), "Behaviour", "both", "behaviour")
MOOD_TEXT = """\
+++
key = "pathway-mood"
+++
## Overview
A mood loop [ref2020].

## Step 1
Role: Starts here [ref2020].
Disorders: Low mood.
Drugs: A pill [ref2020].

## Step 2
Role: Relays.
Disorders: None known.

## Step 3
Role: Ends in the alpha gyrus.
Disorders: Worry.
"""


@pytest.fixture
def mood_atlas(tmp_path):
    meshes = {101: tetra(), 102: tetra(20.0), 110: tetra(40.0)}
    root = write_package(tmp_path / "mood", tiny_structures(), meshes=meshes,
                         pathways=[LOOP, MOOD], references=REFERENCES,
                         entries={"en": {"pathway-mood": MOOD_TEXT}})
    return Atlas.open(root)


def test_old_records_read_with_the_default_kind(tmp_path):
    root = write_package(tmp_path / "pkg", tiny_structures())
    record = {key: value for key, value in LOOP.to_json().items()
              if key not in ("group", "sides", "texts")}
    (root / "pathways.json").write_text(json.dumps([record]))
    pathway = Atlas.open(root).pathways[0]
    assert (pathway.group, pathway.sides, pathway.texts) == (DEFAULT_GROUP, "one", "lesion")


@pytest.mark.parametrize("change", [{"sides": "three"}, {"texts": "poems"}])
def test_unknown_route_kinds_fail_to_open(tmp_path, change):
    root = write_package(tmp_path / "pkg", tiny_structures())
    (root / "pathways.json").write_text(json.dumps([{**LOOP.to_json(), **change}]))
    with pytest.raises(AtlasError):
        Atlas.open(root)


def test_both_sides_routes_light_left_right_and_midline(mood_atlas):
    assert resolve_both(mood_atlas, Station((100,))) == (101, 102)
    assert resolve_both(mood_atlas, Station((110,))) == (110,)
    assert resolve_both(mood_atlas, Relay("x")) == ()
    assert lights(mood_atlas, MOOD, "right") == [(110,), (), (101, 102)]
    assert lights(mood_atlas, LOOP, "left")[0] == (101,)


def test_behaviour_texts_have_disorders_and_drugs(mood_atlas):
    text = mood_atlas.pathway_text("mood", "left")
    assert text.steps[0] == StepText("Starts here [1].", "", "Low mood.", "A pill [1].")
    assert text.steps[1].drugs == ""


@pytest.mark.parametrize(("body", "problems"), [
    ("Role: A.\nDisorders: B.\nDrugs: C.", []),
    ("Role: A.\nDisorders: B.", []),
    ("Role: A.\nDisorders: B.\nDrugs: C.\nDrugs: D.",
     ["needs at most one 'Drugs:' line, found 2"]),
    ("Role: A.\nLesion: B.",
     ["needs one 'Disorders:' line, found 0", "unexpected line 'Lesion: B.'"]),
    ("Role: A.\nDisorders: B.\nDrugs:", ["'Drugs:' line is empty"]),
])
def test_behaviour_step_problems(body, problems):
    assert step_problems(body, "behaviour") == problems


def test_lesion_style_refuses_disorder_lines():
    assert step_problems("Role: A.\nLesion: B.\nDisorders: C.") == [
        "unexpected line 'Disorders: C.'"]


def test_side_words_lists_every_braced_word():
    assert side_words("the {same} side and {left}") == ["same", "left"]
