import numpy as np

from brain_tui.atlas import build_procedural_regions
from brain_tui.present import HalfBlockEncoder
from brain_tui.renderer import project_centroids, render_scene_fast
from brain_tui.scene import SceneState


def cell_colors(strip):
    out = []
    for seg in strip:
        fg, bg = seg.style.color.triplet, seg.style.bgcolor.triplet
        out += [(tuple(fg), tuple(bg))] * len(seg.text)
    return out


def test_encoder_round_trips_pixels():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 4, size=(6, 5, 3), dtype=np.uint8) * 60
    strips, changed = HalfBlockEncoder().encode(img)
    assert len(strips) == 3 and changed == [0, 1, 2]
    for row, strip in enumerate(strips):
        assert strip.cell_length == 5
        for col, (fg, bg) in enumerate(cell_colors(strip)):
            assert fg == tuple(img[row * 2, col])
            assert bg == tuple(img[row * 2 + 1, col])


def test_encoder_merges_runs_of_equal_cells():
    img = np.zeros((4, 50, 3), dtype=np.uint8)
    strips, _ = HalfBlockEncoder().encode(img)
    assert all(len(list(s)) == 1 for s in strips)


def test_encoder_reports_only_changed_rows():
    enc = HalfBlockEncoder()
    img = np.zeros((8, 10, 3), dtype=np.uint8)
    enc.encode(img)
    img2 = img.copy()
    img2[5, 3] = (255, 0, 0)
    _, changed = enc.encode(img2)
    assert changed == [2]
    _, changed = enc.encode(np.zeros((10, 10, 3), dtype=np.uint8))
    assert changed == list(range(5))


def test_isolated_exploded_region_renders_where_its_label_is():
    scene = SceneState(regions=build_procedural_regions())
    scene.set_explode(1.0)
    scene.isolate("hippocampus_L")
    _, idbuf, _ = render_scene_fast(scene, 160, 120)
    sx, sy, _ = project_centroids(scene, 160, 120)["hippocampus_L"]
    x, y = int(sx), int(sy)
    assert (idbuf[max(0, y - 3):y + 4, max(0, x - 3):x + 4] >= 0).any()
