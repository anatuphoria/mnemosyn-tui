import pytest

from atlas_package import TOY_SOURCE, tetra, write_package
from cns_atlas.model import Structure

BLUE, RED, GREEN, GREY = (90, 140, 230), (230, 120, 120), (120, 200, 140), (170, 170, 180)

TOY_REFERENCES = """\
[ref2020]
authors = "Author A"
title = "A reference"
year = 2020
url = "https://example.org/ref"
"""

TOY_ENTRY = """\
+++
id = 100
status = "draft"
+++
## Function
Does alpha things [ref2020].
"""


def sided(concept: int, acronym: str, name: str, rgb, kind: str, side: str, xref: str):
    offset = 1 if side == "left" else 2
    path = (1, 2, concept, concept + offset)
    return Structure(concept + offset, f"{acronym}-{side[0].upper()}", f"{name} ({side})", path,
                     rgb, kind=kind, side=side, concept=concept, synonyms=(f"{side} {name}",),
                     xrefs={"toy": xref}, sources=("toy",))


def toy_structures() -> list[Structure]:
    return [
        Structure(1, "CNS", "Central nervous system", (1,), GREY, kind="group"),
        Structure(2, "Br", "Brain", (1, 2), GREY, kind="group"),
        Structure(100, "ALG", "Alpha gyrus", (1, 2, 100), BLUE, synonyms=("first gyrus",)),
        sided(100, "ALG", "Alpha gyrus", BLUE, "region", "left", "Alpha_L"),
        sided(100, "ALG", "Alpha gyrus", BLUE, "region", "right", "Alpha_R"),
        Structure(110, "BEN", "Beta nucleus", (1, 2, 110), RED, kind="nucleus"),
        sided(110, "BEN", "Beta nucleus", RED, "nucleus", "left", "Beta_L"),
        sided(110, "BEN", "Beta nucleus", RED, "nucleus", "right", "Beta_R"),
        Structure(120, "GAM", "Gamma body", (1, 2, 120), GREEN, kind="nucleus", side="midline",
                  xrefs={"toy": "Gamma"}, sources=("toy",)),
    ]


@pytest.fixture(scope="session")
def toy_atlas(tmp_path_factory):
    meshes = {sid: tetra(20.0 * n) for n, sid in enumerate((101, 102, 111, 112, 120))}
    return write_package(tmp_path_factory.mktemp("toy") / "toy", toy_structures(),
                         meshes=meshes, lods=("lod0", "lod1", "lod2"), sources=(TOY_SOURCE,),
                         entries={"en": {"100": TOY_ENTRY}}, references=TOY_REFERENCES)
