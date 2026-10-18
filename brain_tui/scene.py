from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import numpy as np


@dataclass
class Mesh:
    verts: np.ndarray
    faces: np.ndarray
    color: tuple[int, int, int] = (120, 140, 200)

    @property
    def centroid(self) -> np.ndarray:
        return self.verts.mean(axis=0).astype(np.float32)

    @property
    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        return self.verts.min(axis=0), self.verts.max(axis=0)


@dataclass
class Region:
    id: str
    name: str
    system: str
    color: tuple[int, int, int]
    mesh: Mesh
    base_center: np.ndarray
    visible: bool = True
    opacity: float = 1.0
    move: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))


@dataclass
class Camera:
    yaw: float = -0.6
    pitch: float = 0.35
    distance: float = 3.2
    pan_x: float = 0.0
    pan_y: float = 0.0
    fov: float = math.radians(40.0)
    t_yaw: float | None = None
    t_pitch: float | None = None
    t_dist: float | None = None
    t_panx: float | None = None
    t_pany: float | None = None

    def __post_init__(self) -> None:
        self._sync_targets()

    def _sync_targets(self) -> None:
        if self.t_yaw is None:
            self.t_yaw, self.t_pitch = self.yaw, self.pitch
            self.t_dist = self.distance
            self.t_panx, self.t_pany = self.pan_x, self.pan_y

    def rotate(self, dyaw: float, dpitch: float) -> None:
        self._sync_targets()
        self.t_yaw += dyaw
        self.t_pitch = max(-1.45, min(1.45, self.t_pitch + dpitch))

    def zoom(self, factor: float) -> None:
        self._sync_targets()
        self.t_dist = max(1.2, min(8.0, self.t_dist * factor))

    def pan(self, dx: float, dy: float) -> None:
        self._sync_targets()
        self.t_panx += dx
        self.t_pany += dy

    def nudge_yaw(self, d: float) -> None:
        self._sync_targets()
        self.t_yaw += d

    def fly_to(self, yaw: float, pitch: float, distance: float,
               pan_x: float, pan_y: float) -> None:
        self._sync_targets()
        turns = round((self.t_yaw - yaw) / math.tau)
        self.t_yaw = yaw + turns * math.tau
        self.t_pitch, self.t_dist = pitch, distance
        self.t_panx, self.t_pany = pan_x, pan_y

    def _target_deltas(self) -> tuple[float, ...]:
        return (self.t_yaw - self.yaw, self.t_pitch - self.pitch, self.t_dist - self.distance,
                self.t_panx - self.pan_x, self.t_pany - self.pan_y)

    def ease_pending(self, eps: float = 2e-4) -> bool:
        return self.t_yaw is not None and max(map(abs, self._target_deltas())) >= eps

    def ease(self, factor: float = 0.35, eps: float = 2e-4) -> bool:
        self._sync_targets()
        dy, dp, dd, dx, dy2 = self._target_deltas()
        if max(abs(dy), abs(dp), abs(dd), abs(dx), abs(dy2)) < eps:
            self.yaw, self.pitch = self.t_yaw, self.t_pitch
            self.distance, self.pan_x, self.pan_y = self.t_dist, self.t_panx, self.t_pany
            return False
        self.yaw += dy * factor
        self.pitch += dp * factor
        self.distance += dd * factor
        self.pan_x += dx * factor
        self.pan_y += dy2 * factor
        return True

    def view_matrix(self) -> np.ndarray:
        """World -> camera, right-handed, camera looks down -Z."""
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
        R = Rx @ Ry
        t = np.array([self.pan_x, self.pan_y, -self.distance], dtype=np.float32)
        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = R
        M[:3, 3] = t
        return M


@dataclass
class ClipPlane:
    enabled: bool = False
    axis: str = "x"
    value: float = 0.0


@dataclass
class SceneState:
    regions: dict[str, Region] = field(default_factory=dict)
    camera: Camera = field(default_factory=Camera)
    explode_t: float = 0.0
    explode_scale: float = 0.9
    clip: ClipPlane = field(default_factory=ClipPlane)
    selected_id: str | None = None
    isolated_id: str | None = None
    show_labels: bool = True
    auto_rotate: bool = False

    @property
    def brain_center(self) -> np.ndarray:
        if not self.regions:
            return np.zeros(3, dtype=np.float32)
        centers = np.stack([r.base_center for r in self.regions.values()])
        return centers.mean(axis=0).astype(np.float32)

    def snapshot(self) -> SceneState:
        """Copy that is safe to hand to the render thread."""
        return replace(
            self,
            regions={rid: replace(r, move=r.move.copy()) for rid, r in self.regions.items()},
            camera=replace(self.camera),
            clip=replace(self.clip),
        )

    def visible_regions(self) -> list[Region]:
        out = []
        for r in self.regions.values():
            if not r.visible:
                continue
            if self.isolated_id and r.id != self.isolated_id:
                continue
            out.append(r)
        return out

    def world_centers(self, regions: list[Region]) -> np.ndarray:
        """World centres: base + explode offset + user move."""
        if not regions:
            return np.zeros((0, 3), dtype=np.float32)
        base = np.stack([r.base_center for r in regions]).astype(np.float32)
        move = np.stack([r.move for r in regions]).astype(np.float32)
        away = base - self.brain_center
        dist = np.linalg.norm(away, axis=1, keepdims=True)
        dirs = np.where(dist > 1e-6, away / np.maximum(dist, 1e-6), 0.0)
        return (base + dirs * (self.explode_t * self.explode_scale) + move).astype(np.float32)

    def region_world_center(self, r: Region) -> np.ndarray:
        return self.world_centers([r])[0]

    def select(self, rid: str | None) -> None:
        self.selected_id = rid

    def toggle_visibility(self, rid: str) -> None:
        self.regions[rid].visible = not self.regions[rid].visible

    def isolate(self, rid: str | None) -> None:
        self.isolated_id = rid

    def unhide_all(self) -> None:
        for r in self.regions.values():
            r.visible = True
        self.isolated_id = None

    def set_explode(self, t: float) -> None:
        self.explode_t = max(0.0, min(1.0, t))

    def move_selected(self, dx: float, dy: float, dz: float) -> None:
        if self.selected_id and self.selected_id in self.regions:
            self.regions[self.selected_id].move += np.array([dx, dy, dz], dtype=np.float32)
