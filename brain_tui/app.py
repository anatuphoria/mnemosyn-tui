from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
from pathlib import Path

from rich.markup import escape
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.geometry import Region as ScreenRegion
from textual.message import Message
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import Footer, Header, Input, Label, Static, Tree
from textual.widgets.tree import TreeNode

from cns_atlas import DISCLAIMER

from .atlas import build_procedural_regions, open_atlas, regions_from_atlas
from .knowledge import KnowledgeBase

DEFAULT_ATLAS = Path(__file__).resolve().parent.parent / "atlases" / "cns"
from .pipeline import Frame, RenderRequest, RenderWorker
from .renderer import pick_region, _FULL_BUDGET, _DRAG_BUDGET
from .scene import Region, SceneState

MOVE_STEP = 0.08


def detect_graphics_protocol() -> str:
    env = os.environ.get("PAR_TERM_GRAPHICS", "").strip().lower()
    if env in ("kitty", "sixel", "iterm2", "halfblock"):
        return env
    try:
        from par_textual_image import detect_protocol
        return detect_protocol()
    except Exception:
        pass
    term = (os.environ.get("TERM", "") + os.environ.get("TERM_PROGRAM", "")).lower()
    if "kitty" in term or "wezterm" in term or "ghostty" in term:
        return "kitty"
    return "halfblock"


class FrameReady(Message):
    def __init__(self, frame: Frame) -> None:
        super().__init__()
        self.frame = frame


class RenderFailed(Message):
    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error


class Viewport(Widget):

    can_focus = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame: Frame | None = None
        self._lines: list[Strip] = []
        self._dragging = False
        self._last: tuple[float, float] | None = None
        self._down_at: tuple[float, float] | None = None
        self._down_time = 0.0

    def show_frame(self, frame: Frame) -> None:
        self.frame = frame
        if not frame.strips:
            return
        if len(self._lines) != len(frame.strips):
            self._lines = [Strip.blank(frame.cols)] * len(frame.strips)
        for y in frame.changed_rows:
            self._lines[y] = frame.strips[y]
        width = self.size.width
        self.refresh(*(ScreenRegion(0, y, width, 1) for y in frame.changed_rows))

    def render_line(self, y: int) -> Strip:
        width = self.size.width
        if y >= len(self._lines):
            return Strip.blank(width)
        line = self._lines[y]
        return line if line.cell_length == width else line.adjust_cell_length(width)

    def on_resize(self, e: events.Resize) -> None:
        if isinstance(self.app, BrainApp):
            self.app.need_render()

    def on_mouse_down(self, e: events.MouseDown) -> None:
        self.focus()
        self._dragging = True
        self._last = (e.x, e.y)
        self._down_at = (e.x, e.y)
        self._down_time = time.time()
        self.capture_mouse(True)

    def on_mouse_up(self, e: events.MouseUp) -> None:
        self.capture_mouse(False)
        was_click = (
            self._down_at is not None
            and abs(e.x - self._down_at[0]) < 2
            and abs(e.y - self._down_at[1]) < 2
            and time.time() - self._down_time < 0.6
        )
        self._dragging = False
        self._last = None
        offset = e.get_content_offset(self)
        if was_click and offset is not None and isinstance(self.app, BrainApp):
            self.app.pick_at(offset.x, offset.y)

    async def on_mouse_move(self, e: events.MouseMove) -> None:
        if not self._dragging or self._last is None:
            return
        dx = e.x - self._last[0]
        dy = e.y - self._last[1]
        self._last = (e.x, e.y)
        app = self.app
        if isinstance(app, BrainApp):
            if e.shift:
                app.scene.camera.pan(dx * 0.02, -dy * 0.02)
            else:
                app.scene.camera.rotate(dx * 0.08, dy * 0.06)
            app._touch_camera()

    async def on_mouse_scroll_down(self, e: events.MouseScrollDown) -> None:
        app = self.app
        if isinstance(app, BrainApp):
            app.scene.camera.zoom(1.12)
            app._touch_camera()

    async def on_mouse_scroll_up(self, e: events.MouseScrollUp) -> None:
        app = self.app
        if isinstance(app, BrainApp):
            app.scene.camera.zoom(0.89)
            app._touch_camera()


class BrainApp(App):
    CSS = """
    #main { height: 1fr; }
    #left { width: 28%; min-width: 24; border-right: solid $primary; }
    #center { width: 1fr; }
    #right { width: 30%; min-width: 26; border-left: solid $primary; }
    #viewport { height: 1fr; border: solid $accent; }
    #labels { height: auto; max-height: 8; color: $text-muted; }
    #status { height: 1; background: $surface; color: $text; }
    #search { height: 3; display: none; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("left,h", "arrow(-1, 0)", "Rot ◀"),
        Binding("right,l", "arrow(1, 0)", "Rot ▶"),
        Binding("up,k", "arrow(0, 1)", "Rot ▲"),
        Binding("down,j", "arrow(0, -1)", "Rot ▼"),
        Binding("plus,=", "zoom_in", "Zoom+"),
        Binding("minus,_", "zoom_out", "Zoom-"),
        Binding("w", "pan_up", "Pan↑"),
        Binding("s", "pan_down", "Pan↓"),
        Binding("a", "pan_left", "Pan←"),
        Binding("d", "pan_right", "Pan→"),
        Binding("r", "toggle_spin", "Spin"),
        Binding("i", "isolate", "Isolate"),
        Binding("u", "unhide", "Unhide"),
        Binding("H", "hide_sel", "Hide"),
        Binding("x", "explode_more", "Explode+"),
        Binding("X", "explode_less", "Explode-"),
        Binding("1", "slice_axial", "Axial"),
        Binding("2", "slice_coronal", "Coronal"),
        Binding("3", "slice_sagittal", "Sagittal"),
        Binding("comma", "clip_down", "Clip-"),
        Binding("period", "clip_up", "Clip+"),
        Binding("L", "toggle_labels", "Labels"),
        Binding("b", "bookmark", "Bookmark"),
        Binding("B", "restore_bookmark", "Restore"),
        Binding("m", "move_mode", "Move"),
        # priority: otherwise Textual's focus-next binding takes Tab
        Binding("tab", "cycle_region(1)", "Next region", priority=True),
        Binding("shift+tab", "cycle_region(-1)", "Prev region", show=False, priority=True),
        Binding("slash", "focus_search", "Search"),
        Binding("question_mark", "help", "Help"),
        Binding("z", "quiz", "Quiz"),
        Binding("o", "toggle_hifi", "HiFi"),
    ]

    def __init__(self, atlas: str | None = None, views_file: str | None = None):
        super().__init__()
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.atlas = open_atlas(atlas or DEFAULT_ATLAS)
        regions = regions_from_atlas(self.atlas) if self.atlas else build_procedural_regions()
        self.scene = SceneState(regions=regions)
        self.kb = KnowledgeBase(self.atlas)
        self.protocol = detect_graphics_protocol()
        self.hifi = False
        self._move_mode = False
        self._quiz_id: str | None = None
        self._kitty_widget = None
        self._views_file = views_file or os.path.join(base, "views.json")
        self.region_nodes: dict[str, TreeNode] = {}
        self._dirty = True
        self._in_flight = False
        self._interacting = False
        self._budget = _FULL_BUDGET
        self._crisp_timer = None
        self._search_restore_id = None
        self._gc_frozen = False
        self._worker = RenderWorker(
            deliver=lambda frame: self.post_message(FrameReady(frame)),
            fail=lambda error: self.post_message(RenderFailed(error)),
        )
        if self.protocol in ("kitty", "sixel", "iterm2"):
            try:
                from par_textual_image import TerminalImage
                self._kitty_widget = TerminalImage()
            except Exception:
                self._kitty_widget = None
                self.protocol = "halfblock"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="  /  search regions - e.g. hippocampus, thalamus, cerebellum …", id="search")
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield Label("🧠 REGIONS (Tab cycles, click selects)")
                yield Tree("Brain", id="tree")
            with Vertical(id="center"):
                if self._kitty_widget is not None:
                    yield self._kitty_widget
                yield Viewport(id="viewport")
                yield Static("", id="labels")
            with Vertical(id="right"):
                yield Label("🔬 INSPECTOR")
                yield Static("Select a region.", id="inspector")
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        for rid, r in self.scene.regions.items():
            if r.name.startswith("Hippocampus"):
                self.scene.select(rid)
                break
        self._build_tree()
        self.update_inspector()
        self._worker.start()
        self.set_interval(1 / 30, self._tick)
        self.query_one("#viewport", Viewport).focus()

    def on_unmount(self) -> None:
        self._worker.stop()

    def _region_label(self, r: Region) -> str:
        mark = "👁" if r.visible else "🚫"
        sel = "▶ " if r.id == self.scene.selected_id else ""
        return f"{sel}{mark} {r.name}"

    def _build_tree(self, filt: str = "") -> None:
        tree = self.query_one("#tree", Tree)
        tree.clear()
        self.region_nodes = {}
        root = tree.root
        groups: dict[str, list] = {}
        matches = set(self._matching_region_ids(filt)) if filt.strip() else None
        for rid, r in self.scene.regions.items():
            if matches is not None and rid not in matches and filt.lower() not in r.system.lower():
                continue
            groups.setdefault(r.system, []).append((rid, r))
        for sys in sorted(groups):
            node = root.add(sys, expand=True)
            for rid, r in sorted(groups[sys], key=lambda t: t[1].name):
                self.region_nodes[rid] = node.add_leaf(self._region_label(r), data=rid)
        root.expand()

    def _refresh_tree_labels(self, *rids: str | None) -> None:
        for rid in rids or self.region_nodes:
            node = self.region_nodes.get(rid or "")
            if node is not None:
                node.set_label(self._region_label(self.scene.regions[rid]))

    def _select(self, rid: str | None) -> None:
        prev = self.scene.selected_id
        self.scene.select(rid)
        self._quiz_id = None
        self._refresh_tree_labels(prev, rid)
        node = self.region_nodes.get(rid or "")
        if node is not None:
            self.query_one("#tree", Tree).move_cursor(node)
        self.update_inspector()
        self.need_render()

    def on_tree_node_selected(self, e: Tree.NodeSelected) -> None:
        if e.node.data and e.node.data != self.scene.selected_id:
            self._select(str(e.node.data))

    def update_inspector(self) -> None:
        box = self.query_one("#inspector", Static)
        r = self.scene.regions.get(self.scene.selected_id or "")
        if r is None:
            box.update("No selection.\nClick the brain, press Tab, or / search.")
            return
        k = self.kb.get(r.id)
        iso = " (ISOLATED)" if self.scene.isolated_id == r.id else ""
        draft = "[yellow]Draft: not yet reviewed[/]\n" if k["draft"] else ""
        references = f"[dim]{escape(k['references'])}[/]\n\n" if k["references"] else ""
        box.update(
            f"[bold]{r.name}[/]{iso}\n{draft}"
            f"system: {r.system}   color: #{r.color[0]:02x}{r.color[1]:02x}{r.color[2]:02x}\n"
            f"visible: {r.visible}   move: ({r.move[0]:+.2f},{r.move[1]:+.2f},{r.move[2]:+.2f})\n\n"
            f"[bold]Function:[/] {escape(k['function'])}\n\n"
            f"[bold]Connections:[/] {escape(k['connections'])}\n\n"
            f"[bold]Clinical:[/] {escape(k['disorders'])}\n\n"
            f"{references}"
            f"[dim]I isolate · H hide · U unhide · M move-mode + arrows · X explode[/]\n"
            f"[dim red]{DISCLAIMER}[/]"
        )

    def on_input_changed(self, e: Input.Changed) -> None:
        if e.input.id == "search":
            self._build_tree(e.value)
            hits = self._matching_region_ids(e.value)
            if hits and e.value.strip():
                self._select(hits[0])

    def _matching_region_ids(self, text: str) -> list[str]:
        """Atlas search (names, acronyms, synonyms, atlas labels) or names when procedural."""
        if self.atlas is None:
            by_name = {r.name: rid for rid, r in self.scene.regions.items()}
            names = self.kb.search(text, list(by_name), limit=len(by_name))
            return [by_name[name] for name in names]
        found: list[str] = []
        for structure_id in self.atlas.search(text, limit=None):
            for s in self.atlas.structures[self.atlas.subtree(structure_id)]:
                rid = str(s.id)
                if rid in self.scene.regions and rid not in found:
                    found.append(rid)
        return found

    def need_render(self) -> None:
        self._dirty = True

    def render_idle(self) -> bool:
        return not self._dirty and not self._in_flight

    def _touch_camera(self) -> None:
        if not self._interacting:
            self._interacting = True
            self._budget = _DRAG_BUDGET
        try:
            if self._crisp_timer is not None:
                self._crisp_timer.stop()
        except Exception:
            pass
        self._crisp_timer = self.set_timer(0.3, self._go_crisp)
        self._dirty = True

    def _go_crisp(self) -> None:
        self._interacting = False
        self._budget = _FULL_BUDGET
        self._dirty = True

    def _tick(self) -> None:
        if self.scene.auto_rotate:
            self.scene.camera.nudge_yaw(0.015)
            self._interacting = True
            self._budget = _DRAG_BUDGET
            self._dirty = True
        elif self.scene.camera.ease():
            self._dirty = True
        if self._dirty:
            self._request_frame()

    def _request_frame(self) -> None:
        """Submit the newest scene state; at most one frame in flight."""
        if self._in_flight:
            return
        vp = self.query_one("#viewport", Viewport)
        cols, rows = vp.size
        if cols < 2 or rows < 2:
            return
        halfblock = self._kitty_widget is None
        if halfblock:
            w_px, h_px = cols, rows * 2
        else:
            w_px, h_px = max(200, cols * 6), max(150, rows * 9)
        self._worker.submit(RenderRequest(
            scene=self.scene.snapshot(), cols=cols, rows=rows, width=w_px, height=h_px,
            hifi=self.hifi, budget=self._budget, halfblock=halfblock))
        self._in_flight = True
        self._dirty = False

    def on_frame_ready(self, message: FrameReady) -> None:
        self._in_flight = False
        frame = message.frame
        if not self._gc_frozen:
            # Strips form ref cycles; frequent small GC passes avoid ~70ms hitches
            gc.freeze()
            gc.set_threshold(10_000, 1, 1)
            self._gc_frozen = True
        self.query_one("#viewport", Viewport).show_frame(frame)
        if frame.image is not None and self._kitty_widget is not None:
            try:
                self._kitty_widget.set_image(frame.image)
            except Exception as exc:
                self.notify(f"Graphics protocol failed ({exc}); using half-blocks")
                self._kitty_widget.display = False
                self._kitty_widget = None
                self.protocol = "halfblock"
                self._dirty = True
        self.query_one("#labels", Static).update(frame.labels)
        self._update_status(frame)
        if self._dirty:
            self._request_frame()

    def on_render_failed(self, message: RenderFailed) -> None:
        self._in_flight = False
        self.notify(f"Render failed: {message.error!r}", severity="error")

    def _update_status(self, frame: Frame) -> None:
        n = len(self.scene.visible_regions())
        sel = self.scene.regions.get(self.scene.selected_id or "")
        seln = sel.name if sel else "-"
        clip = "off" if not self.scene.clip.enabled else f"{self.scene.clip.axis}={self.scene.clip.value:+.2f}"
        self.query_one("#status", Static).update(
            f"frame {frame.total_ms:3.0f}ms (render {frame.render_ms:.0f} + encode {frame.encode_ms:.0f}) │ "
            f"gfx:{self.protocol} │ regions:{n}/{len(self.scene.regions)} │ "
            f"sel:{seln} │ explode:{self.scene.explode_t:.2f} │ clip:{clip} │ "
            f"{'MOVE-MODE arrows+PgUp/Dn' if self._move_mode else 'R spin'} │ {'HIFI' if self.hifi else 'FAST'} │ ? help"
        )

    def pick_at(self, cx: int, cy: int) -> None:
        frame = self.query_one("#viewport", Viewport).frame
        if frame is None:
            return
        h, w = frame.idbuf.shape
        px = int((cx + 0.5) * w / frame.cols)
        py = int((cy + 0.25) * h / frame.rows)
        rid = pick_region(self.scene, frame.idbuf, frame.region_ids, px, py)
        if rid:
            self._select(rid)

    def action_arrow(self, x: int, y: int):
        if self._move_mode:
            self._move_selected(x * MOVE_STEP, y * MOVE_STEP, 0)
        else:
            self.scene.camera.rotate(x * 0.15, -y * 0.12); self._touch_camera()
    def _move_selected(self, dx: float, dy: float, dz: float):
        self.scene.move_selected(dx, dy, dz); self.need_render(); self.update_inspector()
    def action_zoom_in(self): self.scene.camera.zoom(0.88); self._touch_camera()
    def action_zoom_out(self): self.scene.camera.zoom(1.14); self._touch_camera()
    def _pan_or_move(self, dx: float, dy: float):
        if self._move_mode: self._move_selected(dx, dy, 0)
        else: self.scene.camera.pan(dx, dy); self._touch_camera()
    def action_pan_up(self): self._pan_or_move(0, MOVE_STEP)
    def action_pan_down(self): self._pan_or_move(0, -MOVE_STEP)
    def action_pan_left(self): self._pan_or_move(-MOVE_STEP, 0)
    def action_pan_right(self): self._pan_or_move(MOVE_STEP, 0)
    def action_toggle_spin(self):
        self.scene.auto_rotate = not self.scene.auto_rotate
        if self.scene.auto_rotate:
            self._interacting = True
            self._budget = _DRAG_BUDGET
            self._dirty = True
        else:
            self._go_crisp()
    def action_isolate(self):
        if self.scene.selected_id:
            self.scene.isolate(None if self.scene.isolated_id == self.scene.selected_id else self.scene.selected_id)
            self.need_render(); self.update_inspector()
    def action_unhide(self): self.scene.unhide_all(); self.need_render(); self._refresh_tree_labels(); self.update_inspector()
    def action_hide_sel(self):
        if self.scene.selected_id:
            self.scene.toggle_visibility(self.scene.selected_id)
            self.need_render(); self._refresh_tree_labels(self.scene.selected_id); self.update_inspector()
    def action_explode_more(self): self.scene.set_explode(self.scene.explode_t + 0.1); self.need_render()
    def action_explode_less(self): self.scene.set_explode(self.scene.explode_t - 0.1); self.need_render()
    def _cycle_clip(self, axis: str):
        c = self.scene.clip
        if c.enabled and c.axis == axis:
            c.enabled = False
        else:
            c.enabled = True; c.axis = axis; c.value = 0.0
        self.need_render()
    def action_slice_axial(self): self._cycle_clip("y")
    def action_slice_coronal(self): self._cycle_clip("z")
    def action_slice_sagittal(self): self._cycle_clip("x")
    def action_clip_down(self):
        self.scene.clip.enabled = True
        self.scene.clip.value -= 0.1; self.need_render()
    def action_clip_up(self):
        self.scene.clip.enabled = True
        self.scene.clip.value += 0.1; self.need_render()
    def action_toggle_labels(self): self.scene.show_labels = not self.scene.show_labels; self.need_render()
    def action_move_mode(self): self._move_mode = not self._move_mode; self.need_render()
    def action_cycle_region(self, step: int):
        ids = list(self.scene.regions.keys())
        if not ids: return
        i = ids.index(self.scene.selected_id) if self.scene.selected_id in ids else -1
        self._select(ids[(i + step) % len(ids)])
    def action_focus_search(self):
        self._search_restore_id = self.scene.selected_id
        inp = self.query_one("#search", Input)
        inp.display = True
        inp.focus()
        inp.select_all()

    def _dismiss_search(self, clear: bool) -> None:
        inp = self.query_one("#search", Input)
        if clear:
            inp.value = ""
            self._build_tree()
        inp.display = False
        self.query_one("#viewport", Viewport).focus()

    def on_input_submitted(self, e: Input.Submitted) -> None:
        if e.input.id == "search":
            self._dismiss_search(clear=False)
            self.need_render()
    def action_help(self):
        self.query_one("#inspector", Static).update(
            "[bold]Keys[/]\n"
            "drag rotate · wheel zoom · click select · shift+drag pan\n"
            "arrows/HJKL rotate · +/- zoom · WASD pan · R spin\n"
            "I isolate · H hide · U unhide · X/x explode\n"
            "1/2/3 slice planes · ,/. move plane · L labels\n"
            "M move-mode: arrows/WASD move region, PgUp/Dn depth, Esc exits\n"
            "Tab/Shift+Tab next/prev · / search-bar (Enter commit, Esc cancel)\n"
            "B save view · Shift+B restore · Z quiz (Z again reveals) · Q quit"
        )

    @staticmethod
    def _view_number(name: str) -> int:
        try:
            return int(name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return 0

    def _load_views(self) -> dict:
        try:
            with open(self._views_file) as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def action_bookmark(self):
        cam = self.scene.camera
        data = {"camera": {"yaw": cam.yaw, "pitch": cam.pitch, "distance": cam.distance,
                           "pan_x": cam.pan_x, "pan_y": cam.pan_y},
                "explode": self.scene.explode_t, "selected": self.scene.selected_id,
                "hidden": [r.id for r in self.scene.regions.values() if not r.visible]}
        try:
            views = self._load_views()
            name = f"view_{max(map(self._view_number, views), default=0) + 1}"
            views[name] = data
            with open(self._views_file, "w") as f:
                json.dump(views, f, indent=2)
            self.notify(f"Saved {name} to views.json")
        except Exception as e:
            self.notify(f"Save failed: {e}")

    def action_restore_bookmark(self):
        try:
            views = self._load_views()
            last = views[max(views, key=self._view_number)]
            c = last["camera"]
            self.scene.camera.fly_to(c["yaw"], c["pitch"], c["distance"], c["pan_x"], c["pan_y"])
            self.scene.set_explode(last.get("explode", 0.0))
            for r in self.scene.regions.values():
                r.visible = r.id not in last.get("hidden", [])
            self._refresh_tree_labels()
            self._select(last.get("selected"))
            self.notify("View restored")
        except Exception as e:
            self.notify(f"No saved view: {e}")

    def action_toggle_hifi(self):
        self.hifi = not self.hifi
        self.notify(f"{'HiFi supersampled' if self.hifi else 'Fast splat renderer'}")
        self.need_render()

    def action_quiz(self):
        if self._quiz_id is not None:
            self._quiz_id = None
            self.update_inspector()
            return
        ids = list(self.scene.regions.keys())
        if not ids: return
        rid = random.choice(ids)
        self._select(rid)
        self._quiz_id = rid
        r = self.scene.regions[rid]
        self.query_one("#inspector", Static).update(
            f"[bold]QUIZ:[/] What does [yellow]{r.name}[/] do?\n\n"
            f"[dim]Hint - system: {r.system}[/]\n\n"
            f"Press [bold]Z[/] to reveal the answer."
        )

    def on_key(self, e: events.Key) -> None:
        if e.key == "escape" and self.focused is self.query_one("#search", Input):
            self._dismiss_search(clear=True)
            if getattr(self, "_search_restore_id", None) in self.scene.regions:
                self._select(self._search_restore_id)
            e.stop()
            return
        if e.key == "escape":
            self._move_mode = False
            self.need_render()
        elif e.key == "pageup" and self._move_mode:
            self._move_selected(0, 0, MOVE_STEP)
        elif e.key == "pagedown" and self._move_mode:
            self._move_selected(0, 0, -MOVE_STEP)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from cns_atlas import Atlas, AtlasError

    p = argparse.ArgumentParser(description="3D TUI Brain Model")
    p.add_argument("--atlas", default=None, help="atlas package directory (default: atlases/cns)")
    args = p.parse_args(argv)
    try:
        if args.atlas is not None:
            Atlas.open(args.atlas)
        app = BrainApp(atlas=args.atlas)
    except AtlasError as exc:
        print(f"{exc}\nBuild the atlas first: python -m tools.build_atlas", file=sys.stderr)
        return 1
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
