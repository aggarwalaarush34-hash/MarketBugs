"""The frozen evaluation contract.

`freeze` selects the market configuration on the development split only and
writes configs/frozen.json with the selected config, the sha256 of both data
files and a hash of every source file. `check` refuses a held-out run if any
of these changed afterwards, so the held-out numbers come from exactly the
code and config that were frozen.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

from .data import ROOT, path_for, sha256

FROZEN = ROOT / "configs" / "frozen.json"
SRC = ROOT / "src" / "marketdebug"


class FreezeViolation(Exception):
    pass


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(SRC.rglob("*.py")):
        h.update(str(p.relative_to(SRC)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def write(config: dict, selection: dict) -> dict:
    rec = {
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config": config,
        "selection": selection,
        "data_sha256": {s: sha256(path_for(s)) for s in ("dev", "test")},
        "code_sha256": code_hash(),
    }
    FROZEN.parent.mkdir(exist_ok=True)
    FROZEN.write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def load() -> dict:
    if not FROZEN.exists():
        raise FreezeViolation("configs/frozen.json missing: run `python -m marketdebug freeze` first")
    return json.loads(FROZEN.read_text())


def check(rec: dict | None = None) -> dict:
    rec = rec or load()
    problems = []
    for s, h in rec["data_sha256"].items():
        if sha256(path_for(s)) != h:
            problems.append(f"data/{path_for(s).name} changed since freeze")
    if code_hash() != rec["code_sha256"]:
        problems.append("source code changed since freeze (re-run freeze on dev, or revert)")
    if problems:
        raise FreezeViolation("; ".join(problems))
    return rec
