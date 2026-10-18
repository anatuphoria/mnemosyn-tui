# Mnemosyn TUI

A terminal viewer for the Mnemosyn atlas of the human central nervous system. It draws the
atlas in 3D inside the terminal, lets you search, select, slice and quiz yourself, and shows
each structure's description with its references.

The desktop app and the tool that builds the atlas data live in
[Mnemosyn](https://github.com/anatuphoria/mnemosyn). This viewer reads the atlas packages that
tool builds.

> **Disclaimer.** This atlas is a personal study project. It was not built or reviewed by a
> neuroanatomy specialist and may contain errors. It is not for diagnosis or treatment.
> Corrections and feedback are welcome.

## Quickstart

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/mnemosyn-tui --atlas /path/to/mnemosyn/atlases/cns
```

`mnemosyn-tui [--atlas PATH]` reads an atlas package. Without `--atlas` it looks for
`atlases/cns` next to this folder and falls back to a 36-region procedural brain when none is
there; a `--atlas` path that does not exist is an error. Build the package with the Mnemosyn
repository (`python -m tools.build_atlas`) or point `--atlas` at one you already have.

Descriptions show numbered citations with their references, and drafts are marked. Letters
are lowercase unless written with `Shift+`.

| Action | Keys |
|---|---|
| Rotate | arrows / `hjkl` or mouse-drag |
| Zoom / Pan | `+`/`-`, `wasd` or wheel / shift+drag |
| Spin | `r` |
| Select | click, `Tab` / `Shift+Tab`, `/` search (`Enter` commits, `Esc` cancels) |
| Move selected | `m` move-mode: arrows or `wasd`, `PgUp`/`PgDn` depth, `Esc` exits |
| Isolate / Hide / Unhide | `i` / `Shift+H` / `u` |
| Explode | `x` / `Shift+X` |
| Slice axial/coronal/sagittal | `1` / `2` / `3`, `,`/`.` move plane |
| Labels | `Shift+L` |
| Bookmark save/restore | `b` / `Shift+B` (saved in `views.json`) |
| HiFi stills / Quiz | `o` / `z` (`z` again reveals the answer) |
| Help | `?` |

## Layout

```
brain_tui/   the terminal viewer
cns_atlas/   reader for Mnemosyn atlas packages (shared format with the desktop app)
tests/       pytest suite; the toy atlas is written by hand, no build tool needed
```

`cns_atlas` is a copy of the reader in the Mnemosyn repository. When the atlas package format
changes there, copy the reader here too.

## Tests

```bash
.venv/bin/python -m pytest
```

## Licence

GPL-3.0-or-later, see `LICENSE`. The atlas data a package contains carries its own licences,
listed in that package.
