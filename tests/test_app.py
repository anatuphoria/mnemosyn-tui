import asyncio
import json
import time

import numpy as np

from textual.widgets import Input

from brain_tui.app import BrainApp, Viewport, main
from brain_tui.knowledge import KnowledgeBase
from cns_atlas import Atlas


def run(coro):
    return asyncio.run(coro)


async def wait_until(pilot, cond, timeout: float = 10.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return
        await pilot.pause(0.05)
    raise AssertionError("condition not met before timeout")


async def settle(pilot, app) -> None:
    vp = app.query_one("#viewport", Viewport)

    def current() -> bool:
        frame = vp.frame
        return frame is not None and (frame.cols, frame.rows) == tuple(vp.size)

    await wait_until(pilot, lambda: not app.scene.camera.ease_pending() and app.render_idle()
                     and current())


def make_app(tmp_path) -> BrainApp:
    return BrainApp(atlas=str(tmp_path / "no-atlas"), views_file=str(tmp_path / "views.json"))


def inspector_text(app) -> str:
    return str(app.query_one("#inspector").render())


def test_viewport_paints_at_content_resolution(tmp_path):
    async def go():
        app = make_app(tmp_path)
        async with app.run_test(size=(140, 44)) as pilot:
            vp = app.query_one("#viewport", Viewport)
            await wait_until(pilot, lambda: vp.frame is not None)
            await settle(pilot, app)
            w, h = vp.size
            assert vp.frame.idbuf.shape == (h * 2, w)
            assert len(vp.frame.strips) == h
            assert (vp.frame.idbuf >= 0).mean() > 0.05
    run(go())


def test_click_selects_region_under_cursor(tmp_path):
    async def go():
        app = make_app(tmp_path)
        async with app.run_test(size=(140, 44)) as pilot:
            vp = app.query_one("#viewport", Viewport)
            await wait_until(pilot, lambda: vp.frame is not None)
            await settle(pilot, app)
            idbuf, ids = vp.frame.idbuf, vp.frame.region_ids
            ys, xs = np.nonzero(idbuf[0::2] >= 0)
            target = None
            for cy, cx in zip(ys, xs):
                rid = ids[idbuf[cy * 2, cx]]
                if rid != app.scene.selected_id:
                    target = (int(cx), int(cy), rid)
                    break
            assert target is not None
            cx, cy, rid = target
            gx, gy = vp.gutter.top_left
            await pilot.click("#viewport", offset=(cx + gx, cy + gy))
            await pilot.pause(0.1)
            assert app.scene.selected_id == rid
    run(go())


def test_bookmark_restore_returns_to_saved_view(tmp_path):
    async def go():
        app = make_app(tmp_path)
        async with app.run_test(size=(140, 44)) as pilot:
            await settle(pilot, app)
            saved_yaw = app.scene.camera.yaw
            await pilot.press("b")
            for _ in range(6):
                await pilot.press("right")
            await settle(pilot, app)
            assert abs(app.scene.camera.yaw - saved_yaw) > 0.5
            await pilot.press("B")
            await settle(pilot, app)
            assert abs(app.scene.camera.yaw - saved_yaw) < 1e-3
    run(go())


def test_bookmark_restore_uses_newest_view(tmp_path):
    views = {f"view_{i}": {"camera": {"yaw": i / 10, "pitch": 0.1, "distance": 3.0,
                                      "pan_x": 0.0, "pan_y": 0.0}}
             for i in range(1, 12)}
    (tmp_path / "views.json").write_text(json.dumps(views))

    async def go():
        app = make_app(tmp_path)
        async with app.run_test(size=(140, 44)) as pilot:
            await pilot.press("B")
            await settle(pilot, app)
            assert abs(app.scene.camera.yaw - 1.1) < 1e-3
            await pilot.press("b")
            assert "view_12" in json.loads((tmp_path / "views.json").read_text())
    run(go())


def test_tab_cycles_regions(tmp_path):
    async def go():
        app = make_app(tmp_path)
        async with app.run_test(size=(140, 44)) as pilot:
            before = app.scene.selected_id
            await pilot.press("tab")
            assert app.scene.selected_id != before
    run(go())


def test_tab_does_not_rebuild_tree(tmp_path):
    async def go():
        app = make_app(tmp_path)
        async with app.run_test(size=(140, 44)) as pilot:
            tree = app.query_one("#tree")
            first_group = tree.root.children[0]
            await pilot.press("tab")
            await pilot.press("tab")
            assert tree.root.children[0] is first_group
            label = str(app.region_nodes[app.scene.selected_id].label)
            assert label.startswith("▶")
    run(go())


def test_move_mode_arrows_move_selected_region(tmp_path):
    async def go():
        app = make_app(tmp_path)
        async with app.run_test(size=(140, 44)) as pilot:
            region = app.scene.regions[app.scene.selected_id]
            yaw = app.scene.camera.t_yaw
            await pilot.press("m")
            await pilot.press("right")
            assert region.move[0] > 0
            assert app.scene.camera.t_yaw == yaw
            await pilot.press("escape")
            await pilot.press("right")
            assert app.scene.camera.t_yaw != yaw
    run(go())


def test_quiz_hides_answer_until_revealed(tmp_path):
    async def go():
        app = make_app(tmp_path)
        async with app.run_test(size=(140, 44)) as pilot:
            await pilot.press("z")
            region = app.scene.regions[app.scene.selected_id]
            answer = app.kb.get(region.id)["function"]
            assert "QUIZ" in inspector_text(app)
            assert answer not in inspector_text(app)
            await pilot.press("z")
            assert answer in inspector_text(app)
    run(go())


def test_atlas_regions_and_descriptions(tmp_path, toy_atlas):
    async def go():
        app = BrainApp(atlas=str(toy_atlas), views_file=str(tmp_path / "views.json"))
        async with app.run_test(size=(140, 44)):
            assert set(app.scene.regions) == {"101", "102", "111", "112", "120"}
            assert app.scene.regions["101"].system == "Brain"
            app._select("101")
            assert "Does alpha things [1]." in inspector_text(app)
            app._select("111")
            assert "No description yet." in inspector_text(app)
    run(go())


def test_search_uses_atlas_labels_acronyms_and_concepts(tmp_path, toy_atlas):
    async def go():
        app = BrainApp(atlas=str(toy_atlas), views_file=str(tmp_path / "views.json"))
        async with app.run_test(size=(140, 44)) as pilot:
            search = app.query_one("#search", Input)
            for text, expected in (("alpha_l", "101"), ("ALG-R", "102"), ("beta nucleus", "111")):
                search.value = text
                await pilot.pause()
                assert app.scene.selected_id == expected, text
    run(go())


def test_descriptions_number_citations_and_flag_drafts(toy_atlas):
    entry = KnowledgeBase(Atlas.open(toy_atlas)).get("101")
    assert entry["function"] == "Does alpha things [1]."
    assert entry["draft"] == "yes"
    assert entry["references"].startswith("[1] Author A (2020). A reference.")


def test_cli_rejects_a_missing_atlas_path(tmp_path, capsys):
    assert main(["--atlas", str(tmp_path / "missing")]) == 1
    assert "not found" in capsys.readouterr().err


def test_cli_reports_a_broken_atlas(tmp_path, capsys):
    root = tmp_path / "broken"
    root.mkdir()
    (root / "structures.json").write_text("[]")
    (root / "metadata.json").write_text("{}")
    assert main(["--atlas", str(root)]) == 1
    assert "format_version" in capsys.readouterr().err


def test_cli_without_atlas_uses_the_built_in_shapes(monkeypatch, tmp_path):
    import brain_tui.app as app_module

    started = {}
    monkeypatch.setattr(app_module, "DEFAULT_ATLAS", tmp_path / "none")
    monkeypatch.setattr(app_module.BrainApp, "run", lambda self: started.setdefault("ok", self.atlas))
    assert main([]) == 0 and started == {"ok": None}

def test_cli_rejects_a_folder_that_is_not_an_atlas(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(BrainApp, "run", lambda self: None)
    folder = tmp_path / "not-an-atlas"
    folder.mkdir()
    assert main(["--atlas", str(folder)]) == 1
    assert "metadata.json" in capsys.readouterr().err
