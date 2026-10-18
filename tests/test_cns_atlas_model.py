import numpy as np

from cns_atlas.meshes import MeshData, MeshLod, write_lod
from cns_atlas.model import (Collection, Crossing, Pathway, Relay, Source, Station, Structure,
                             step_from_json, step_to_json)


def test_structure_round_trips_through_json():
    s = Structure(101, "ALG-L", "Alpha gyrus (left)", (1, 2, 100, 101), (90, 140, 230),
                  kind="region", side="left", concept=100, synonyms=("left alpha gyrus",),
                  xrefs={"toy": "Alpha_L"}, sources=("toy",))
    data = s.to_json()
    assert data["structure_id_path"] == [1, 2, 100, 101]
    assert Structure.from_json(data) == s
    assert s.parent == 100 and s.depth == 3


def test_structure_without_concept_omits_the_key():
    root = Structure(1, "CNS", "Central nervous system", (1,), (170, 170, 180), kind="group")
    assert "concept" not in root.to_json()
    assert root.parent is None and root.depth == 0
    assert Structure.from_json(root.to_json()).concept is None


def test_source_redistribution_must_be_literally_true():
    data = {"id": "x", "title": "T", "authors": "A", "year": 2020, "version": "1",
            "licence": "CC0 1.0", "redistribution": "yes"}
    assert Source.from_json(data).redistribution is False
    data["redistribution"] = True
    source = Source.from_json(data)
    assert source.redistribution is True
    assert source.citation() == "A (2020). T, version 1."


def test_collection_and_pathway_round_trip():
    c = Collection("limbic-system", "Limbic system", (1350, 1400))
    p = Pathway("pain", "Pain and temperature", "side of the body",
                (Relay("Dorsal horn"), Crossing("Anterior white commissure"),
                 Station((3260,), "opposite"), Station((1840, 1700), "opposite"),
                 Station((3320,), "midline")))
    assert Collection.from_json(c.to_json()) == c
    assert Pathway.from_json(p.to_json()) == p
    assert p.to_json()["steps"][:2] == [{"relay": "Dorsal horn", "side": "same"},
                                        {"cross": "Anterior white commissure"}]
    both = Pathway("mood", "Mood", "raphe", (Station((2200,)),), "Behaviour", "both", "behaviour")
    assert Pathway.from_json(both.to_json()) == both
    assert both.to_json()["group"] == "Behaviour"


def tetra(offset):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32) + offset
    normals = np.eye(4, 3, dtype=np.float16)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.uint32)
    return MeshData(vertices, normals, faces)


def test_mesh_lod_round_trip(tmp_path):
    write_lod(tmp_path / "lod0", {7: tetra(7), 3: tetra(3)})
    lod = MeshLod(tmp_path / "lod0")
    assert lod.ids() == (3, 7)
    assert 7 in lod and 8 not in lod
    m = lod.get(7)
    assert m.vertices.dtype == np.float32 and m.normals.dtype == np.float16
    assert m.faces.dtype == np.uint32
    np.testing.assert_array_equal(m.vertices, tetra(7).vertices)
    np.testing.assert_array_equal(m.faces, tetra(7).faces)
    assert lod.get(8) is None


def test_empty_mesh_lod(tmp_path):
    write_lod(tmp_path / "lod0", {})
    lod = MeshLod(tmp_path / "lod0")
    assert lod.ids() == () and lod.get(1) is None


def test_a_crossing_station_round_trips():
    station = Station((3320,), "midline", crossing=True)
    assert step_from_json(step_to_json(station)) == station
    assert "crossing" not in step_to_json(Station((3320,), "midline"))


def test_source_provides_round_trips():
    from cns_atlas.model import Source

    source = Source("x", "T", "A B", 2020, "1", "CC0", True, provides="Spinal cord")
    assert Source.from_json(source.to_json()) == source
    data = source.to_json()
    del data["provides"]
    assert Source.from_json(data).provides == ""
