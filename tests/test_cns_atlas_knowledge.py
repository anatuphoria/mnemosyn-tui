import codecs

import pytest

from cns_atlas.knowledge import KnowledgeError, load_references, parse_entry, resolve

ENTRY = """+++
id = 1350
status = "reviewed"
+++
## Function
Forms new memories [textbook2018]. Also navigation [paper2001][textbook2018].

## Clinical
See [this page](https://example.org) for details [paper2001].
"""

REFERENCES = """
[textbook2018]
authors = "Author A, Author B"
title = "Example Textbook"
edition = "6th"
year = 2018
publisher = "Example Press"
isbn = "9780000000000"

[paper2001]
authors = "Author C"
title = "An example paper"
journal = "Journal of Examples"
volume = "12(3)"
pages = "100-110"
year = 2001
doi = "10.0000/example"
"""


def test_parse_sections_status_and_citation_order():
    entry = parse_entry(ENTRY, "1350.md")
    assert entry.target == 1350 and entry.status == "reviewed"
    assert list(entry.sections) == ["Function", "Clinical"]
    assert entry.citations == ("textbook2018", "paper2001")


def test_markdown_links_are_not_citations():
    entry = parse_entry(ENTRY, "1350.md")
    assert "this page" not in entry.citations


def test_windows_line_endings_parse_the_same():
    assert parse_entry(ENTRY.replace("\n", "\r\n"), "x.md") == parse_entry(ENTRY, "x.md")


def test_byte_order_mark_parses_the_same():
    bom = codecs.BOM_UTF8.decode()
    assert parse_entry(bom + ENTRY, "x.md") == parse_entry(ENTRY, "x.md")


def test_key_target_and_default_status():
    entry = parse_entry('+++\nkey = "limbic-system"\n+++\n## Function\nText [a2020].', "k.md")
    assert entry.target == "limbic-system" and entry.status == "draft"


@pytest.mark.parametrize("text, message", [
    ("## Function\nText.", r"missing '\+\+\+' header"),
    ("+++\nid = 1\n## Function\nText.", "not closed"),
    ('+++\nid = 1\nkey = "x"\n+++\n', "exactly one of"),
    ('+++\nid = "1"\n+++\n', "must be an integer"),
    ('+++\nid = 1\nstatus = "final"\n+++\n', "status must be one of"),
    ("+++\nid = 1\n+++\n## Anatomy\nText.", "unknown section 'Anatomy'"),
    ("+++\nid = 1\n+++\nloose text\n## Function\nText.", "text before the first section"),
    ("+++\nid = = 1\n+++\n", "invalid header"),
])
def test_malformed_entries_are_rejected(text, message):
    with pytest.raises(KnowledgeError, match=message):
        parse_entry(text, "bad.md")


def test_references_resolve_and_number(tmp_path):
    path = tmp_path / "references.toml"
    path.write_text(REFERENCES)
    refs = load_references(path)
    entry = resolve(parse_entry(ENTRY, "1350.md"), refs)
    assert [r.key for r in entry.references] == ["textbook2018", "paper2001"]
    assert entry.numbered(entry.sections["Function"]) == "Forms new memories [1]. Also navigation [2][1]."


def test_reference_citations_and_links(tmp_path):
    path = tmp_path / "references.toml"
    path.write_text(REFERENCES)
    refs = load_references(path)
    book, paper = refs["textbook2018"], refs["paper2001"]
    assert book.citation() == ("Author A, Author B (2018). Example Textbook. 6th edition. "
                               "Example Press. ISBN 9780000000000.")
    assert paper.citation() == ("Author C (2001). An example paper. Journal of Examples 12(3): "
                                "100-110. doi:10.0000/example")
    assert paper.link == "https://doi.org/10.0000/example"
    assert book.has_locator and paper.has_locator


def test_reference_problems(tmp_path):
    path = tmp_path / "references.toml"
    path.write_text('[x2020]\nauthors = "A"\ntitle = "T"\n')
    with pytest.raises(KnowledgeError, match="missing"):
        load_references(path)
    path.write_text('[x2020]\nauthors = "A"\ntitle = "T"\nyear = 2020\npagez = "1"\n')
    with pytest.raises(KnowledgeError, match="unknown fields"):
        load_references(path)
    path.write_text('[x2020]\nauthors = "A"\ntitle = "T"\nyear = 2020\n')
    assert not load_references(path)["x2020"].has_locator


PATHWAY_ENTRY = """+++
key = "pathway-visual"
+++
## Overview
A [a2020].
## Step 1
Role: B.
Lesion: C.
## Step 2
Role: D.
Lesion: E.
"""


def test_pathway_entries_take_an_overview_and_numbered_steps():
    entry = parse_entry(PATHWAY_ENTRY, "p.md")
    assert list(entry.sections) == ["Overview", "Step 1", "Step 2"]
    assert entry.sections["Step 2"] == "Role: D.\nLesion: E."


@pytest.mark.parametrize(("text", "message"), [
    ("+++\nid = 1350\n+++\n## Step 1\nRole: A.\n", "unknown section 'Step 1'"),
    ('+++\nkey = "pathway-x"\n+++\n## Function\nA.\n', "unknown section 'Function'"),
    ('+++\nkey = "pathway-x"\n+++\n## Step 0\nA.\n', "unknown section 'Step 0'"),
    ('+++\nkey = "pathway-x"\n+++\n## Step 01\nA.\n', "unknown section 'Step 01'"),
    ("+++\nid = 1350\n+++\n## Function\nA.\n## Function\nB.\n",
     "section 'Function' appears twice"),
    ('+++\nkey = "pathway-x"\n+++\n## Step 2\nA.\n## Step 2\nB.\n',
     "section 'Step 2' appears twice"),
])
def test_sections_are_checked_by_kind_of_entry(text, message):
    with pytest.raises(KnowledgeError, match=message):
        parse_entry(text, "x.md")
