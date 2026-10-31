from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def ensure_out(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_history(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                fieldnames.append(name)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
