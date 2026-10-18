from __future__ import annotations

from pathlib import Path

import numpy as np


class TractLayer:
    def __init__(self, folder: Path, shape, affine) -> None:
        try:
            ids = np.load(folder / "ids.npy")
            offsets = np.load(folder / "offsets.npy")
            self._voxels = np.load(folder / "voxels.npy", mmap_mode="r")
        except (OSError, ValueError) as exc:
            raise ValueError(f"tract layer is unreadable ({exc})") from exc
        if (len(offsets) != len(ids) + 1 or offsets[0] != 0 or np.any(np.diff(offsets) < 0)
                or offsets[-1] != len(self._voxels)):
            raise ValueError("tract offsets do not match the voxel list")
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("tract ids are repeated")
        steps = np.diff(np.asarray(self._voxels, dtype=np.int64))
        within = np.ones(len(steps), dtype=bool)
        starts = offsets[1:-1]
        within[starts[(starts > 0) & (starts < len(self._voxels))] - 1] = False
        if np.any(steps[within] <= 0):
            raise ValueError("tract voxels are not sorted within each tract")
        self.shape = tuple(int(n) for n in shape[:3])
        if len(self._voxels) and int(np.max(self._voxels)) >= int(np.prod(self.shape)):
            raise ValueError("tract voxels lie outside the label grid")
        self.affine = np.asarray(affine, dtype=float)
        self._world_to_voxel = np.linalg.inv(self.affine)
        self.ids = tuple(int(i) for i in ids)
        self._spans = {sid: (int(offsets[n]), int(offsets[n + 1]))
                       for n, sid in enumerate(self.ids)}
        self._by_size = sorted(self.ids, key=lambda sid: (self.size(sid), sid))

    def size(self, structure_id: int) -> int:
        start, end = self._spans.get(structure_id, (0, 0))
        return end - start

    def voxels(self, structure_id: int) -> np.ndarray:
        start, end = self._spans.get(structure_id, (0, 0))
        return self._voxels[start:end]

    def flat_index(self, point) -> int | None:
        world = np.append(np.asarray(point, dtype=float), 1.0)
        ijk = np.rint(self._world_to_voxel @ world)[:3].astype(int)
        if np.any(ijk < 0) or np.any(ijk >= np.array(self.shape)):
            return None
        return int(np.ravel_multi_index(tuple(ijk), self.shape))

    def tracts_at(self, point) -> tuple[int, ...]:
        flat = self.flat_index(point)
        if flat is None:
            return ()
        found = []
        for structure_id in self._by_size:
            voxels = self.voxels(structure_id)
            n = int(np.searchsorted(voxels, flat))
            if n < len(voxels) and int(voxels[n]) == flat:
                found.append(structure_id)
        return tuple(found)
