from __future__ import annotations

from dataclasses import dataclass, field

FORMAT_VERSION = 1
KINDS = ("group", "region", "nucleus", "tract", "ventricle", "nerve-root")
SIDES = ("left", "right", "midline", "none")
STEP_SIDES = ("same", "opposite", "midline")
PATHWAY_SIDES = ("one", "both")
TEXT_STYLES = ("lesion", "behaviour")
DEFAULT_GROUP = "Sensory and motor"


@dataclass(frozen=True)
class Structure:
    id: int
    acronym: str
    name: str
    structure_id_path: tuple[int, ...]
    rgb_triplet: tuple[int, int, int]
    kind: str = "region"
    side: str = "none"
    concept: int | None = None
    synonyms: tuple[str, ...] = ()
    xrefs: dict[str, str] = field(default_factory=dict, hash=False)
    sources: tuple[str, ...] = ()

    @property
    def parent(self) -> int | None:
        return self.structure_id_path[-2] if len(self.structure_id_path) > 1 else None

    @property
    def depth(self) -> int:
        return len(self.structure_id_path) - 1

    def to_json(self) -> dict:
        data = {
            "id": self.id,
            "acronym": self.acronym,
            "name": self.name,
            "structure_id_path": list(self.structure_id_path),
            "rgb_triplet": list(self.rgb_triplet),
            "kind": self.kind,
            "side": self.side,
            "synonyms": list(self.synonyms),
            "xrefs": dict(sorted(self.xrefs.items())),
            "sources": list(self.sources),
        }
        if self.concept is not None:
            data["concept"] = self.concept
        return data

    @classmethod
    def from_json(cls, data: dict) -> Structure:
        concept = data.get("concept")
        return cls(
            id=int(data["id"]),
            acronym=str(data["acronym"]),
            name=str(data["name"]),
            structure_id_path=tuple(int(i) for i in data["structure_id_path"]),
            rgb_triplet=tuple(int(c) for c in data["rgb_triplet"]),
            kind=str(data.get("kind", "region")),
            side=str(data.get("side", "none")),
            concept=int(concept) if concept is not None else None,
            synonyms=tuple(str(s) for s in data.get("synonyms", ())),
            xrefs={str(k): str(v) for k, v in data.get("xrefs", {}).items()},
            sources=tuple(str(s) for s in data.get("sources", ())),
        )


@dataclass(frozen=True)
class Source:
    id: str
    title: str
    authors: str
    year: int
    version: str
    licence: str
    redistribution: bool
    doi: str = ""
    url: str = ""
    provides: str = ""

    def citation(self) -> str:
        return f"{self.authors} ({self.year}). {self.title}, version {self.version}."

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "version": self.version,
            "doi": self.doi,
            "url": self.url,
            "licence": self.licence,
            "redistribution": self.redistribution,
            "provides": self.provides,
        }

    @classmethod
    def from_json(cls, data: dict) -> Source:
        return cls(
            id=str(data["id"]),
            title=str(data["title"]),
            authors=str(data["authors"]),
            year=int(data["year"]),
            version=str(data["version"]),
            licence=str(data.get("licence", "")),
            redistribution=data.get("redistribution") is True,
            doi=str(data.get("doi", "")),
            url=str(data.get("url", "")),
            provides=str(data.get("provides", "")),
        )


@dataclass(frozen=True)
class Collection:
    key: str
    name: str
    members: tuple[int, ...]

    def to_json(self) -> dict:
        return {"key": self.key, "name": self.name, "members": list(self.members)}

    @classmethod
    def from_json(cls, data: dict) -> Collection:
        return cls(str(data["key"]), str(data["name"]), tuple(int(i) for i in data["members"]))


@dataclass(frozen=True)
class Station:
    structures: tuple[int, ...]
    side: str = "same"
    crossing: bool = False
    fallback: str = ""


@dataclass(frozen=True)
class Crossing:
    name: str


@dataclass(frozen=True)
class Relay:
    name: str
    side: str = "same"


Step = Station | Crossing | Relay


def step_to_json(step: Step) -> dict:
    if isinstance(step, Station):
        data = {"structures": list(step.structures), "side": step.side}
        if step.crossing:
            data["crossing"] = True
        if step.fallback:
            data["fallback"] = step.fallback
        return data
    if isinstance(step, Crossing):
        return {"cross": step.name}
    return {"relay": step.name, "side": step.side}


def step_from_json(data: dict) -> Step:
    kinds = [kind for kind in ("structures", "cross", "relay") if kind in data]
    if len(kinds) != 1:
        raise ValueError(f"step needs exactly one of structures, cross or relay: {data!r}")
    side = str(data.get("side", "same"))
    if side not in STEP_SIDES:
        raise ValueError(f"unknown step side {side!r}")
    if kinds == ["structures"]:
        return Station(tuple(int(i) for i in data["structures"]), side,
                       data.get("crossing") is True, str(data.get("fallback", "")))
    if kinds == ["cross"]:
        return Crossing(str(data["cross"]))
    return Relay(str(data["relay"]), side)


@dataclass(frozen=True)
class Pathway:
    key: str
    name: str
    start: str
    steps: tuple[Step, ...]
    group: str = DEFAULT_GROUP
    sides: str = "one"
    texts: str = "lesion"

    def to_json(self) -> dict:
        return {"key": self.key, "name": self.name, "start": self.start, "group": self.group,
                "sides": self.sides, "texts": self.texts,
                "steps": [step_to_json(step) for step in self.steps]}

    @classmethod
    def from_json(cls, data: dict) -> Pathway:
        sides, texts = str(data.get("sides", "one")), str(data.get("texts", "lesion"))
        if sides not in PATHWAY_SIDES:
            raise ValueError(f"unknown pathway sides {sides!r}")
        if texts not in TEXT_STYLES:
            raise ValueError(f"unknown pathway texts {texts!r}")
        return cls(str(data["key"]), str(data["name"]), str(data["start"]),
                   tuple(step_from_json(step) for step in data["steps"]),
                   str(data.get("group", DEFAULT_GROUP)), sides, texts)
