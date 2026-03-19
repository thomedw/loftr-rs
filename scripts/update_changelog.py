#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

MARKER = "<!-- release-notes -->"
HEADER = """# Changelog

All notable changes to this project will be documented in this file.

<!-- release-notes -->
"""


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: update_changelog.py <changelog> <release-notes>", file=sys.stderr)
        return 2

    changelog_path = Path(sys.argv[1])
    notes_path = Path(sys.argv[2])

    if not notes_path.is_file():
        print(f"release notes file not found: {notes_path}", file=sys.stderr)
        return 1

    notes = notes_path.read_text(encoding="utf-8").strip()
    if not notes:
        print("release notes were empty", file=sys.stderr)
        return 1

    changelog = HEADER if not changelog_path.exists() else changelog_path.read_text(encoding="utf-8")
    if MARKER not in changelog:
        print(f"marker {MARKER!r} not found in {changelog_path}", file=sys.stderr)
        return 1

    first_heading = next((line.strip() for line in notes.splitlines() if line.strip().startswith("## ")), None)
    if first_heading and first_heading in changelog:
        print(f"release heading already present in {changelog_path}: {first_heading}", file=sys.stderr)
        return 1

    marker_index = changelog.index(MARKER) + len(MARKER)
    before = changelog[:marker_index].rstrip()
    after = changelog[marker_index:].lstrip()
    pieces = [before, "", notes]
    if after:
        pieces.extend(["", after])
    changelog_path.write_text("\n".join(pieces).rstrip() + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
