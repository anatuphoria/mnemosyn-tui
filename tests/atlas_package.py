import json
from pathlib import Path

import nibabel as nib
import numpy as np

from cns_atlas.meshes import MeshData, write_lod
from cns_atlas.model import FORMAT_VERSION, Source, Structure


def tetra(offset: float = 0.0) -> MeshData:
    vertices = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=np.float32) + offset
    normals = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float16)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.uint32)
    return MeshData(vertices, normals, faces)


def tiny_structures() -> list[Structure]:
    blue, red, grey = (90, 140, 230), (230, 120, 120), (170, 170, 180)
    return [
        Structure(1, "CNS", "Central nervous system", (1,), grey, kind="group"),
        Structure(2, "Br", "Brain", (1, 2), grey, kind="group"),
        Structure(100, "ALG", "Alpha gyrus", (1, 2, 100), blue, synonyms=("first gyrus",)),
        Structure(101, "ALG-L", "Alpha gyrus (left)", (1, 2, 100, 101), blue, side="left",
                  concept=100, synonyms=("left Alpha gyrus",), xrefs={"toy": "Alpha_L"},
                  sources=("toy",)),
        Structure(102, "ALG-R", "Alpha gyrus (right)", (1, 2, 100, 102), blue, side="right",
                  concept=100, synonyms=("right Alpha gyrus",), xrefs={"toy": "Alpha_R"},
                  sources=("toy",)),
        Structure(110, "BEN", "Beta nucleus", (1, 2, 110), red, kind="nucleus", side="midline",
                  xrefs={"toy": "Beta"}, sources=("toy",)),
    ]


def write_package(root: Path, structures: list[Structure], *, meshes=None, lods=("lod0",),
                  volume=None, affine=None, sources=(), entries=None, references="",
                  format_version=FORMAT_VERSION, template=None, meshes_by_lod=None,
                  pathways=None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    volumes = []
    if volume is not None:
        (root / "volumes").mkdir(exist_ok=True)
        grid = np.eye(4) if affine is None else affine
        nib.save(nib.Nifti1Image(volume.astype(np.uint16), grid), root / "volumes" / "brain.nii.gz")
        volumes.append({"file": "volumes/brain.nii.gz", "space": "MNI152",
                        "affine_to_world": np.eye(4).tolist()})
        if template is not None:
            nib.save(nib.Nifti1Image(template.astype(np.uint8), grid),
                     root / "volumes" / "brain-template.nii.gz")
            volumes.append({"file": "volumes/brain-template.nii.gz", "space": "MNI152",
                            "role": "template", "labels": "brain",
                            "affine_to_world": np.eye(4).tolist()})
    metadata = {"name": "test-atlas", "version": "0", "format_version": format_version,
                "volumes": volumes}
    (root / "metadata.json").write_text(json.dumps(metadata))
    (root / "structures.json").write_text(json.dumps([s.to_json() for s in structures]))
    (root / "sources.json").write_text(json.dumps([s.to_json() for s in sources]))
    (root / "collections.json").write_text("[]")
    (root / "pathways.json").write_text(json.dumps([p.to_json() for p in pathways or ()]))
    if meshes_by_lod is not None:
        for name, level in meshes_by_lod.items():
            write_lod(root / "meshes" / name, level)
    else:
        for name in lods:
            write_lod(root / "meshes" / name, meshes or {})
    knowledge = root / "knowledge"
    knowledge.mkdir(exist_ok=True)
    (knowledge / "references.toml").write_text(references)
    for lang, by_target in (entries or {}).items():
        (knowledge / lang).mkdir(exist_ok=True)
        for target, text in by_target.items():
            (knowledge / lang / f"{target}.md").write_text(text)
    return root


TOY_SOURCE = Source("toy", "Toy labelled atlas", "Test A", 2024, "1", "CC0 1.0", True)
