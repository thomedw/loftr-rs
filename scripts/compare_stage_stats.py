#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def collect_numeric_deltas(prefix: str, left, right, deltas: list[tuple[str, float]]) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(left.keys() & right.keys()):
            collect_numeric_deltas(f"{prefix}.{key}" if prefix else key, left[key], right[key], deltas)
        return
    if isinstance(left, list) and isinstance(right, list):
        for index, (lv, rv) in enumerate(zip(left, right)):
            collect_numeric_deltas(f"{prefix}[{index}]", lv, rv, deltas)
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        deltas.append((prefix, abs(float(left) - float(right))))


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: compare_stage_stats.py <rust.json> <kornia.json>", file=sys.stderr)
        return 2

    rust = load(sys.argv[1])
    kornia = load(sys.argv[2])
    deltas: list[tuple[str, float]] = []
    collect_numeric_deltas("", rust, kornia, deltas)
    deltas.sort(key=lambda item: item[1], reverse=True)

    print("largest numeric deltas:")
    for name, delta in deltas[:20]:
        print(f"{name}: {delta:.6g}")

    if deltas and math.isfinite(deltas[0][1]):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

