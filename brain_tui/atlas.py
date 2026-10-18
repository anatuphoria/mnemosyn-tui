from __future__ import annotations

from pathlib import Path

import numpy as np

from cns_atlas import Atlas

from .scene import Mesh, Region

MM_PER_UNIT = 90.0


def make_ellipsoid(rx: float, ry: float, rz: float, n_lat: int = 8, n_lon: int = 12) -> Mesh:
    verts: list[list[float]] = []
    for i in range(n_lat + 1):
        theta = np.pi * i / n_lat
        for j in range(n_lon):
            phi = 2 * np.pi * j / n_lon
            x = rx * np.sin(theta) * np.cos(phi)
            y = ry * np.cos(theta)
            z = rz * np.sin(theta) * np.sin(phi)
            verts.append([x, y, z])
    verts = np.array(verts, dtype=np.float32)
    faces: list[list[int]] = []
    for i in range(n_lat):
        for j in range(n_lon):
            a = i * n_lon + j
            b = i * n_lon + (j + 1) % n_lon
            c = (i + 1) * n_lon + j
            d = (i + 1) * n_lon + (j + 1) % n_lon
            if i > 0:
                faces.append([a, c, b])
            if i < n_lat - 1:
                faces.append([b, c, d])
    return Mesh(verts=verts, faces=np.array(faces, dtype=np.int32))


def _procedural_defs() -> list[tuple]:
    F = (86, 140, 230)
    P = (86, 200, 170)
    T = (230, 170, 80)
    O = (200, 110, 220)
    S = (235, 120, 120)
    C = (140, 180, 120)
    B = (180, 170, 160)
    defs = [
        ("frontal_sup_L", "Frontal_Sup_L", "frontal", F, (-0.42, 0.45, 0.55), (0.34, 0.38, 0.42)),
        ("frontal_sup_R", "Frontal_Sup_R", "frontal", F, (0.42, 0.45, 0.55), (0.34, 0.38, 0.42)),
        ("frontal_mid_L", "Frontal_Mid_L", "frontal", F, (-0.55, 0.10, 0.55), (0.28, 0.34, 0.40)),
        ("frontal_mid_R", "Frontal_Mid_R", "frontal", F, (0.55, 0.10, 0.55), (0.28, 0.34, 0.40)),
        ("frontal_inf_L", "Frontal_Inf_Tri_L", "frontal", F, (-0.50, -0.20, 0.45), (0.24, 0.26, 0.32)),
        ("frontal_inf_R", "Frontal_Inf_Tri_R", "frontal", F, (0.50, -0.20, 0.45), (0.24, 0.26, 0.32)),
        ("precentral_L", "Precentral_L", "frontal", F, (-0.58, 0.15, 0.05), (0.20, 0.45, 0.28)),
        ("precentral_R", "Precentral_R", "frontal", F, (0.58, 0.15, 0.05), (0.20, 0.45, 0.28)),
        ("postcentral_L", "Postcentral_L", "parietal", P, (-0.60, 0.15, -0.30), (0.20, 0.44, 0.26)),
        ("postcentral_R", "Postcentral_R", "parietal", P, (0.60, 0.15, -0.30), (0.20, 0.44, 0.26)),
        ("parietal_sup_L", "Parietal_Sup_L", "parietal", P, (-0.40, 0.55, -0.35), (0.32, 0.30, 0.36)),
        ("parietal_sup_R", "Parietal_Sup_R", "parietal", P, (0.40, 0.55, -0.35), (0.32, 0.30, 0.36)),
        ("precuneus_L", "Precuneus_L", "parietal", P, (-0.22, 0.35, -0.55), (0.24, 0.32, 0.30)),
        ("precuneus_R", "Precuneus_R", "parietal", P, (0.22, 0.35, -0.55), (0.24, 0.32, 0.30)),
        ("temporal_sup_L", "Temporal_Sup_L", "temporal", T, (-0.62, -0.25, 0.05), (0.22, 0.24, 0.42)),
        ("temporal_sup_R", "Temporal_Sup_R", "temporal", T, (0.62, -0.25, 0.05), (0.22, 0.24, 0.42)),
        ("temporal_mid_L", "Temporal_Mid_L", "temporal", T, (-0.55, -0.50, -0.05), (0.24, 0.22, 0.38)),
        ("temporal_mid_R", "Temporal_Mid_R", "temporal", T, (0.55, -0.50, -0.05), (0.24, 0.22, 0.38)),
        ("occipital_L", "Occipital_Mid_L", "occipital", O, (-0.35, 0.10, -0.80), (0.30, 0.38, 0.30)),
        ("occipital_R", "Occipital_Mid_R", "occipital", O, (0.35, 0.10, -0.80), (0.30, 0.38, 0.30)),
        ("insula_L", "Insula_L", "subcortical", S, (-0.38, 0.0, 0.10), (0.16, 0.20, 0.22)),
        ("insula_R", "Insula_R", "subcortical", S, (0.38, 0.0, 0.10), (0.16, 0.20, 0.22)),
        ("thalamus_L", "Thalamus_L", "subcortical", S, (-0.14, 0.02, -0.10), (0.13, 0.15, 0.16)),
        ("thalamus_R", "Thalamus_R", "subcortical", S, (0.14, 0.02, -0.10), (0.13, 0.15, 0.16)),
        ("hippocampus_L", "Hippocampus_L", "limbic", (235, 150, 150), (-0.22, -0.28, -0.15), (0.11, 0.12, 0.24)),
        ("hippocampus_R", "Hippocampus_R", "limbic", (235, 150, 150), (0.22, -0.28, -0.15), (0.11, 0.12, 0.24)),
        ("amygdala_L", "Amygdala_L", "limbic", (240, 130, 130), (-0.20, -0.18, 0.18), (0.10, 0.11, 0.11)),
        ("amygdala_R", "Amygdala_R", "limbic", (240, 130, 130), (0.20, -0.18, 0.18), (0.10, 0.11, 0.11)),
        ("caudate_L", "Caudate_L", "subcortical", S, (-0.16, 0.22, 0.10), (0.10, 0.16, 0.20)),
        ("caudate_R", "Caudate_R", "subcortical", S, (0.16, 0.22, 0.10), (0.10, 0.16, 0.20)),
        ("cerebellum_L", "Cerebellum_6_L", "cerebellum", C, (-0.32, -0.62, -0.55), (0.30, 0.24, 0.28)),
        ("cerebellum_R", "Cerebellum_6_R", "cerebellum", C, (0.32, -0.62, -0.55), (0.30, 0.24, 0.28)),
        ("vermis", "Vermis_7", "cerebellum", C, (0.0, -0.55, -0.60), (0.14, 0.18, 0.20)),
        ("pons", "Brainstem_Pons", "brainstem", B, (0.0, -0.30, 0.05), (0.16, 0.18, 0.16)),
        ("medulla", "Brainstem_Medulla", "brainstem", B, (0.0, -0.55, -0.05), (0.12, 0.20, 0.13)),
        ("midbrain", "Brainstem_Midbrain", "brainstem", B, (0.0, -0.10, -0.05), (0.13, 0.14, 0.13)),
    ]
    return defs


def build_procedural_regions() -> dict[str, Region]:
    regions: dict[str, Region] = {}
    for rid, name, system, color, center, radii in _procedural_defs():
        mesh = make_ellipsoid(*radii)
        c = np.array(center, dtype=np.float32)
        if abs(float(c[0])) < 0.45 and system not in ("brainstem",):
            c[0] += 0.07 if c[0] >= 0 else -0.07
        regions[rid] = Region(
            id=rid, name=name, system=system, color=tuple(color),
            mesh=mesh, base_center=c,
        )
    return regions


def open_atlas(path) -> Atlas | None:
    root = Path(path)
    return Atlas.open(root) if (root / "structures.json").exists() else None


def _outward(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    if float((v0 * np.cross(v1, v2)).sum()) < 0:
        return faces[:, ::-1].copy()
    return faces


def regions_from_atlas(atlas: Atlas) -> dict[str, Region]:
    regions: dict[str, Region] = {}
    for s in atlas.structures:
        data = atlas.mesh(s.id, lod=1)
        if data is None:
            continue
        world = np.asarray(data.vertices, dtype=np.float32)[:, [0, 2, 1]] / MM_PER_UNIT
        center = world.mean(axis=0).astype(np.float32)
        verts = (world - center).astype(np.float32)
        faces = _outward(verts, np.asarray(data.faces, dtype=np.int32))
        group = atlas.get(s.concept if s.concept is not None else s.id).parent
        system = atlas.get(group).name if group is not None else "unknown"
        regions[str(s.id)] = Region(id=str(s.id), name=s.name, system=system,
                                    color=s.rgb_triplet, mesh=Mesh(verts, faces, s.rgb_triplet),
                                    base_center=center)
    return regions
