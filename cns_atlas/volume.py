from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


class Volume:
    def __init__(self, path: Path, space: str, affine_to_world: np.ndarray) -> None:
        self.path = path
        self.space = space
        self.affine_to_world = np.asarray(affine_to_world, dtype=float)
        self._data: np.ndarray | None = None
        self._world_to_voxel: np.ndarray | None = None
        self._header_affine: np.ndarray | None = None
        self._shape: tuple[int, int, int] | None = None

    def _load(self) -> np.ndarray:
        if self._data is None:
            image = nib.load(self.path)
            self._data = np.asanyarray(image.dataobj)
            self._world_to_voxel = np.linalg.inv(self.affine_to_world @ image.affine)
        return self._data

    def _read_header(self) -> None:
        if self._shape is None:
            image = nib.load(self.path)
            self._header_affine = np.asarray(image.affine, dtype=float)
            self._shape = tuple(int(n) for n in image.shape[:3])

    @property
    def affine(self) -> np.ndarray:
        self._read_header()
        return self.affine_to_world @ self._header_affine

    @property
    def shape(self) -> tuple[int, int, int]:
        self._read_header()
        return self._shape

    @property
    def name(self) -> str:
        return Path(self.path).name.split(".")[0]

    @property
    def data(self) -> np.ndarray:
        return self._load()


class LabelVolume(Volume):
    def label_at(self, point) -> int:
        data = self._load()
        world = np.append(np.asarray(point, dtype=float), 1.0)
        ijk = np.rint(self._world_to_voxel @ world)[:3].astype(int)
        if np.any(ijk < 0) or np.any(ijk >= np.array(data.shape[:3])):
            return 0
        return int(data[tuple(ijk)])


class TemplateVolume(Volume):
    def __init__(self, path: Path, space: str, affine_to_world: np.ndarray, labels: str) -> None:
        super().__init__(path, space, affine_to_world)
        self.labels = labels
