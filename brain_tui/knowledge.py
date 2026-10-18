from __future__ import annotations

from cns_atlas import Atlas

NO_DESCRIPTION = "No description yet."


class KnowledgeBase:
    def __init__(self, atlas: Atlas | None = None) -> None:
        self.atlas = atlas

    def get(self, region_id: str) -> dict[str, str]:
        entry = None
        if self.atlas is not None and region_id.isdigit():
            entry = self.atlas.knowledge(int(region_id))
        sections = entry.sections if entry else {}

        def section(name: str) -> str:
            return entry.numbered(sections[name]) if name in sections else NO_DESCRIPTION

        references = "\n".join(f"[{n}] {ref.citation()}"
                               for n, ref in enumerate(entry.references if entry else (), 1))
        return {"function": section("Function"), "connections": section("Connections"),
                "disorders": section("Clinical"),
                "draft": "yes" if entry is not None and entry.status == "draft" else "",
                "references": references}

    def search(self, query: str, names: list[str], limit: int = 12) -> list[str]:
        q = query.strip().lower()
        if not q:
            return names[:limit]
        scored: list[tuple[int, str]] = []
        for n in names:
            nl = n.lower()
            if nl == q:
                scored.append((0, n))
            elif nl.startswith(q):
                scored.append((1, n))
            elif q in nl:
                scored.append((2, n))
        scored.sort(key=lambda t: (t[0], t[1]))
        return [n for _, n in scored[:limit]]
