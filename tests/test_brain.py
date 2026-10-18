import numpy as np
from brain_tui.atlas import build_procedural_regions
from brain_tui.scene import SceneState, Camera
from brain_tui.renderer import render_scene, render_scene_fast, pick_region
from brain_tui import renderer as R


def test_procedural_builds():
    regs = build_procedural_regions()
    assert len(regs) >= 30
    assert "hippocampus_L" in regs or any("Hippocampus" in r.name for r in regs.values())


def test_render_produces_image_and_ids():
    scene = SceneState(regions=build_procedural_regions())
    img, idbuf, ids = render_scene(scene, 120, 80)
    assert img.size == (120, 80)
    assert idbuf.shape == (80, 120)
    assert (idbuf >= 0).mean() > 0.02
    assert len(ids) == len(scene.visible_regions())


def test_pick_center_hits_brain():
    scene = SceneState(regions=build_procedural_regions())
    img, idbuf, ids = render_scene(scene, 120, 80)
    ys, xs = np.nonzero(idbuf >= 0)
    assert len(xs) > 0
    rid = pick_region(scene, idbuf, ids, int(xs[len(xs) // 2]), int(ys[len(ys) // 2]))
    assert rid in scene.regions


def test_explode_moves_centers():
    scene = SceneState(regions=build_procedural_regions())
    before = {k: scene.region_world_center(v).copy() for k, v in scene.regions.items()}
    scene.set_explode(1.0)
    moved = [np.linalg.norm(scene.region_world_center(v) - before[k]) for k, v in scene.regions.items()]
    assert max(moved) > 0.1


def test_clip_hides_everything_when_extreme():
    scene = SceneState(regions=build_procedural_regions())
    _, idbuf_full, _ = render_scene(scene, 100, 70)
    n_full = (idbuf_full >= 0).sum()
    scene.clip.enabled = True
    scene.clip.axis = "x"
    scene.clip.value = -10.0
    _, idbuf_clip, _ = render_scene(scene, 100, 70)
    assert (idbuf_clip >= 0).sum() < n_full


def test_fast_renderer_covers_and_picks():
    import time
    scene = SceneState(regions=build_procedural_regions())
    t0 = time.time()
    img, idbuf, ids = render_scene_fast(scene, 80, 80)
    dt = time.time() - t0
    assert img.size == (80, 80)
    assert (idbuf >= 0).mean() > 0.15
    assert dt < 0.5
    ys, xs = np.nonzero(idbuf >= 0)
    rid = pick_region(scene, idbuf, ids, int(xs[len(xs) // 2]), int(ys[len(ys) // 2]))
    assert rid in scene.regions


def test_spin_defaults_off():
    assert SceneState(regions=build_procedural_regions()).auto_rotate is False


def test_camera_targets_and_ease():
    cam = Camera()
    cam.rotate(1.0, 0.2)
    assert abs(cam.t_yaw - (cam.yaw + 1.0)) < 1e-6
    assert cam.yaw == -0.6
    for _ in range(200):
        if not cam.ease():
            break
    assert abs(cam.yaw - cam.t_yaw) < 1e-3
    assert abs(cam.pitch - cam.t_pitch) < 1e-3


def test_numba_numpy_parity():
    scene = SceneState(regions=build_procedural_regions())
    scene.camera.yaw = 1.1
    img1, id1, ids1 = render_scene_fast(scene, 100, 70)
    old = R._HAVE_NUMBA
    R._HAVE_NUMBA = False
    try:
        img2, id2, ids2 = render_scene_fast(scene, 100, 70)
    finally:
        R._HAVE_NUMBA = old
    assert ids1 == ids2
    c1 = (id1 >= 0).mean()
    c2 = (id2 >= 0).mean()
    assert abs(c1 - c2) < 0.05
    agree = ((id1 == id2) | ((id1 < 0) & (id2 < 0))).mean()
    assert agree > 0.90
