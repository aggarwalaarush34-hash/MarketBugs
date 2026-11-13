"""Cached Gemma 4 E4B forecasts on MarketBugs: loading and validation.

Each record is one bug: 20 retrieved candidate functions, the indices of the
gold (buggy) candidates, and the frozen Gemma probability agent's forecast
over the candidates for each available subset of specialist notes. Subsets
are bitmasks over AGENTS (bit i = AGENTS[i]); mask 0 is "no notes" and 31 is
"all five notes". Records are produced by scripts/extract_marketbugs.py.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

AGENTS = ("trace", "graph", "spec", "static", "semantic")
FULL = (1 << len(AGENTS)) - 1
REQUIRED_MASKS = (0, *(1 << i for i in range(len(AGENTS))), FULL)
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


class ValidationError(Exception):
    pass


class MissingCoalition(KeyError):
    """The forecast for this subset of notes was never computed."""


def bit(agent: str) -> int:
    return 1 << AGENTS.index(agent)


@dataclass(frozen=True)
class Bug:
    id: str
    repo: str
    stratum: str | None
    gold: tuple[int, ...]
    forecasts: dict[int, tuple[float, ...]]

    @property
    def k(self) -> int:
        return len(self.forecasts[0])

    @property
    def gold_in_pool(self) -> bool:
        return bool(self.gold)

    @property
    def has_full_table(self) -> bool:
        return len(self.forecasts) == 1 << len(AGENTS)

    def forecast(self, mask: int) -> tuple[float, ...]:
        try:
            return self.forecasts[mask]
        except KeyError:
            raise MissingCoalition(f"{self.id}: no forecast for note subset {mask:05b}") from None

    def gold_mass(self, mask: int) -> float:
        p = self.forecast(mask)
        return sum(p[g] for g in self.gold)

    def resolution_weights(self) -> list[float]:
        """Payout weight per candidate: uniform over gold candidates."""
        w = [0.0] * self.k
        for g in self.gold:
            w[g] = 1.0 / len(self.gold)
        return w


def parse(rec: dict) -> Bug:
    return Bug(
        id=rec["id"],
        repo=rec["repo"],
        stratum=rec.get("stratum"),
        gold=tuple(rec["gold"]),
        forecasts={int(m): tuple(p) for m, p in rec["forecasts"].items()},
    )


def validate(bugs: list[Bug], tol: float = 1e-6) -> list[str]:
    """Return a list of problems (empty if the split is valid)."""
    problems: list[str] = []
    seen: set[str] = set()
    for b in bugs:
        if b.id in seen:
            problems.append(f"{b.id}: duplicate id")
        seen.add(b.id)
        for m in REQUIRED_MASKS:
            if m not in b.forecasts:
                problems.append(f"{b.id}: missing required subset {m:05b}")
        if 0 not in b.forecasts:
            continue
        k = b.k
        for m, p in b.forecasts.items():
            if not 0 <= m <= FULL:
                problems.append(f"{b.id}: mask {m} out of range")
            if len(p) != k:
                problems.append(f"{b.id}/{m:05b}: {len(p)} probs, expected {k}")
            if any(x < 0 for x in p):
                problems.append(f"{b.id}/{m:05b}: negative probability")
            if abs(sum(p) - 1) > tol:
                problems.append(f"{b.id}/{m:05b}: probs sum to {sum(p):.8f}")
        for g in b.gold:
            if not 0 <= g < k:
                problems.append(f"{b.id}: gold index {g} out of range")
        if len(set(b.gold)) != len(b.gold):
            problems.append(f"{b.id}: duplicate gold index")
    return problems


def path_for(split: str) -> Path:
    return DATA / f"gemma_forecasts_{split}.jsonl"


def load_split(split: str, path: Path | None = None, check: bool = True) -> list[Bug]:
    path = path or path_for(split)
    bugs = [parse(json.loads(line)) for line in open(path) if line.strip()]
    if check:
        problems = validate(bugs)
        if problems:
            raise ValidationError(f"{path}: {len(problems)} problems, first: {problems[:5]}")
    return bugs


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
