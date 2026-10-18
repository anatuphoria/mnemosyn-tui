import json

import numpy as np
import pytest

from atlas_package import TOY_SOURCE, tetra, tiny_structures, write_package
from cns_atlas import DISCLAIMER, Atlas, AtlasError

REFS = '[ref2020]\nauthors = "Author A"\ntitle = "A reference"\nyear = 2020\nurl = "https://example.org/ref"\n'
ALPHA = "+++\nid = 100\n+++\n## Function\nDoes alpha things [ref2020].\n"
ALPHA_AR = "+++\nid = 100\n+++\n## Function\nArabic text [ref2020].\n"


@pytest.fixture
def package(tmp_path):
    volume = np.zeros((10, 10, 10), dtype=np.uint16)
    volume[2:4, 2:4, 2:4] = 101
    volume[6:8, 6:8, 6:8] = 999
    return write_package(
        tmp_path / "pkg", tiny_structures(),
        meshes={101: tetra(0), 102: tetra(5), 110: tetra(9)},
        volume=volume, sources=(TOY_SOURCE,), references=REFS,
        entries={"en": {100: ALPHA}, "ar": {100: ALPHA_AR}},
    )


def test_open_reads_structures_in_depth_first_order(package):
    atlas = Atlas.open(package)
    assert [s.id for s in atlas.structures] == [1, 2, 100, 101, 102, 110]
    assert atlas.get(101).name == "Alpha gyrus (left)"
    assert 101 in atlas and 999 not in atlas
    assert atlas.sources["toy"].licence == "CC0 1.0"


def test_tree_navigation(package):
    atlas = Atlas.open(package)
    assert atlas.children(2) == (100, 110)
    assert atlas.children(101) == ()
    assert atlas.ancestors(101) == (1, 2, 100)
    assert [s.id for s in atlas.structures[atlas.subtree(100)]] == [100, 101, 102]
    assert [s.id for s in atlas.structures[atlas.subtree(2)]] == [2, 100, 101, 102, 110]
    assert [s.id for s in atlas.structures[atlas.subtree(110)]] == [110]


def test_missing_package_and_files(tmp_path, package):
    with pytest.raises(AtlasError, match="atlas package not found"):
        Atlas.open(tmp_path / "nowhere")
    (package / "structures.json").unlink()
    with pytest.raises(AtlasError, match="structures.json: file is missing"):
        Atlas.open(package)


def test_invalid_json_names_the_file(package):
    (package / "sources.json").write_text("{not json")
    with pytest.raises(AtlasError, match="sources.json: not valid JSON"):
        Atlas.open(package)


def test_other_format_version_is_rejected(package):
    metadata = json.loads((package / "metadata.json").read_text())
    metadata["format_version"] = 2
    (package / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(AtlasError, match="format_version 2 is not supported"):
        Atlas.open(package)


def test_unknown_parent_is_rejected(package):
    data = json.loads((package / "structures.json").read_text())
    data[-1]["structure_id_path"] = [1, 77, 110]
    (package / "structures.json").write_text(json.dumps(data))
    with pytest.raises(AtlasError, match="unknown parent 77"):
        Atlas.open(package)


def test_search_ranks_exact_then_prefix_then_substring(package):
    atlas = Atlas.open(package)
    assert atlas.search("ben") == [110]
    assert atlas.search("alpha")[:1] == [100]
    assert set(atlas.search("alpha")) == {100, 101, 102}
    assert atlas.search("first gyrus") == [100]
    assert atlas.search("") == []
    assert atlas.search("zzz") == []


def test_search_ignores_case_spaces_and_matches_atlas_labels(package):
    atlas = Atlas.open(package)
    assert atlas.search("  ALPHA   gyrus ")[:1] == [100]
    assert atlas.search(" alpha_L ") == [101]
    assert atlas.search("alg-r") == [102]


def test_knowledge_via_concept_with_references(package):
    atlas = Atlas.open(package)
    entry = atlas.knowledge(101)
    assert entry.target == 100
    assert entry.sections["Function"] == "Does alpha things [ref2020]."
    assert [r.key for r in entry.references] == ["ref2020"]
    assert entry.numbered(entry.sections["Function"]) == "Does alpha things [1]."


def test_knowledge_language_fallback_and_missing(package):
    atlas = Atlas.open(package)
    assert "Arabic text" in atlas.knowledge(102, "ar").sections["Function"]
    assert "Does alpha" in atlas.knowledge(102, "de").sections["Function"]
    assert atlas.knowledge(110) is None
    assert atlas.knowledge(999) is None


def test_malformed_knowledge_raises_atlas_error(package):
    (package / "knowledge" / "en" / "110.md").write_text("no header")
    with pytest.raises(AtlasError, match="missing '\\+\\+\\+' header"):
        Atlas.open(package).knowledge(110)


def test_mesh_lookup_and_lod_fallback(package):
    atlas = Atlas.open(package)
    assert atlas.has_mesh(101) and not atlas.has_mesh(100)
    assert atlas.mesh(101, lod=2) is not None
    np.testing.assert_array_equal(atlas.mesh(102).vertices, tetra(5).vertices)
    assert atlas.mesh(100) is None


def test_label_at_inside_outside_and_unknown(package):
    atlas = Atlas.open(package)
    assert atlas.label_at((3.0, 3.0, 3.0)) == 101
    assert atlas.label_at((5.0, 5.0, 5.0)) is None
    assert atlas.label_at((7.0, 7.0, 7.0)) is None
    assert atlas.label_at((500.0, -3.0, 2.0)) is None


def test_disclaimer_text():
    assert DISCLAIMER.startswith("This atlas is a personal study project.")


def test_template_absent_by_default(package):
    atlas = Atlas.open(package)
    assert atlas.template is None
    assert len(atlas.volumes) == 1
    assert atlas.volumes[0].shape == (10, 10, 10)
    assert np.allclose(atlas.volumes[0].affine, np.eye(4))


def test_template_is_loaded_and_not_used_for_labels(tmp_path):
    volume = np.zeros((10, 10, 10), dtype=np.uint16)
    volume[2:4, 2:4, 2:4] = 101
    template = np.full((10, 10, 10), 77, dtype=np.uint8)
    root = write_package(tmp_path / "pkg", tiny_structures(), volume=volume, template=template)
    atlas = Atlas.open(root)
    assert len(atlas.volumes) == 1
    assert atlas.template is not None and atlas.template.labels == "brain"
    assert atlas.template.data.dtype == np.uint8 and int(atlas.template.data[0, 0, 0]) == 77
    assert atlas.template.shape == (10, 10, 10)
    assert atlas.label_at((3.0, 3.0, 3.0)) == 101


def template_package(tmp_path, template_shape=(10, 10, 10), labels="brain"):
    volume = np.zeros((10, 10, 10), dtype=np.uint16)
    volume[2:4, 2:4, 2:4] = 101
    template = np.full(template_shape, 77, dtype=np.uint8)
    root = write_package(tmp_path / "pkg", tiny_structures(), volume=volume, template=template)
    metadata = json.loads((root / "metadata.json").read_text())
    metadata["volumes"][1]["labels"] = labels
    (root / "metadata.json").write_text(json.dumps(metadata))
    return root


def test_a_template_of_another_shape_is_ignored(tmp_path):
    assert Atlas.open(template_package(tmp_path, template_shape=(8, 10, 10))).template is None


def test_a_template_for_another_volume_is_ignored(tmp_path):
    assert Atlas.open(template_package(tmp_path, labels="other")).template is None


def test_a_template_has_no_label_lookup(tmp_path):
    template = Atlas.open(template_package(tmp_path)).template
    assert template is not None and not hasattr(template, "label_at")


def test_local_only_sources_come_from_the_metadata(tmp_path):
    root = write_package(tmp_path / "pkg", tiny_structures())
    assert Atlas.open(root).local_only == ()
    metadata = json.loads((root / "metadata.json").read_text())
    metadata["local_only"] = ["toylocal"]
    (root / "metadata.json").write_text(json.dumps(metadata))
    assert Atlas.open(root).local_only == ("toylocal",)


def edit_json(root, name, change):
    data = json.loads((root / name).read_text())
    (root / name).write_text(json.dumps(change(data)))


@pytest.mark.parametrize(("name", "change", "message"), [
    ("metadata.json", lambda m: [m], "expected a JSON object"),
    ("metadata.json", lambda m: {**m, "format_version": True}, "format_version True"),
    ("metadata.json", lambda m: {**m, "format_version": 1.0}, "format_version 1.0"),
    ("metadata.json", lambda m: {**m, "local_only": "toylocal"}, "local_only must be a list"),
    ("metadata.json", lambda m: {**m, "local_only": None}, "local_only must be a list"),
    ("structures.json", lambda s: s + [s[-1]], r"duplicate structure ids \[110\]"),
    ("structures.json", lambda s: [s[0], s[1], s[2], s[5], s[3], s[4]],
     "structure 101 is not inside its parent's block"),
])
def test_malformed_packages_fail_with_a_clear_message(tmp_path, name, change, message):
    root = write_package(tmp_path / "pkg", tiny_structures())
    edit_json(root, name, change)
    with pytest.raises(AtlasError, match=message):
        Atlas.open(root)


@pytest.mark.parametrize(("name", "content"), [
    ("index.npy", b"not numpy"), ("vertices.npy", b""), ("index.npy", None)])
def test_unreadable_meshes_fail_with_a_clear_message(tmp_path, name, content):
    root = write_package(tmp_path / "pkg", tiny_structures(), meshes={101: tetra()})
    path = root / "meshes" / "lod0" / name
    if content is None:
        np.save(path, np.int64(7))
    else:
        path.write_bytes(content)
    with pytest.raises(AtlasError, match="meshes"):
        Atlas.open(root)
