from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

LODS = ("lod0", "lod1", "lod2")
INDEX_DTYPE = np.dtype([
    ("id", "<u4"),
    ("vertex_start", "<u8"),
    ("vertex_count", "<u4"),
    ("face_start", "<u8"),
    ("face_count", "<u4"),
])


@dataclass(frozen=True)
class MeshData:
    vertices: np.ndarray
    normals: np.ndarray
    faces: np.ndarray


def _stack(arrays: list[np.ndarray], dtype) -> np.ndarray:
    return np.concatenate(arrays).astype(dtype) if arrays else np.zeros((0, 3), dtype=dtype)


def write_lod(directory: Path, meshes: dict[int, MeshData]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    ids = sorted(meshes)
    index = np.zeros(len(ids), dtype=INDEX_DTYPE)
    vertex_start = face_start = 0
    for row, structure_id in enumerate(ids):
        mesh = meshes[structure_id]
        index[row] = (structure_id, vertex_start, len(mesh.vertices), face_start, len(mesh.faces))
        vertex_start += len(mesh.vertices)
        face_start += len(mesh.faces)
    np.save(directory / "vertices.npy", _stack([meshes[i].vertices for i in ids], np.float32))
    np.save(directory / "normals.npy", _stack([meshes[i].normals for i in ids], np.float16))
    np.save(directory / "faces.npy", _stack([meshes[i].faces for i in ids], np.uint32))
    np.save(directory / "index.npy", index)


class MeshLod:
    def __init__(self, directory: Path) -> None:
        index = np.load(directory / "index.npy")
        mode = "r" if len(index) else None
        self.vertices = np.load(directory / "vertices.npy", mmap_mode=mode)
        self.normals = np.load(directory / "normals.npy", mmap_mode=mode)
        self.faces = np.load(directory / "faces.npy", mmap_mode=mode)
        self._rows = {int(row["id"]): row for row in index}

    def __contains__(self, structure_id: object) -> bool:
        return structure_id in self._rows

    def ids(self) -> tuple[int, ...]:
        return tuple(self._rows)

    def get(self, structure_id: int) -> MeshData | None:
        row = self._rows.get(structure_id)
        if row is None:
            return None
        v0, vn = int(row["vertex_start"]), int(row["vertex_count"])
        f0, fn = int(row["face_start"]), int(row["face_count"])
        return MeshData(self.vertices[v0:v0 + vn], self.normals[v0:v0 + vn], self.faces[f0:f0 + fn])
