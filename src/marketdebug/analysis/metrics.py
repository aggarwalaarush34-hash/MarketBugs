"""Per-bug scoring of a forecast over candidates, and aggregate summaries.

Conventions (same as MarketBugs):
  * top-k / reciprocal rank count a retrieval miss (no gold in pool) as 0;
  * NLL and Brier are computed on gold-in-pool bugs only, on the probability
    mass assigned to the gold set;
  * ECE is top-label, 10 equal-width bins, over all bugs.
Ties at the top are broken by lowest candidate index (deterministic).
"""

from __future__ import annotations

import math
from typing import Sequence

EPS = 1e-12


def rank_of_gold(p: Sequence[float], gold: Sequence[int]) -> int | None:
    if not gold:
        return None
    order = sorted(range(len(p)), key=lambda i: (-p[i], i))
    return min(order.index(g) for g in gold) + 1


def score(p: Sequence[float], gold: Sequence[int]) -> dict:
    r = rank_of_gold(p, gold)
    top = min(range(len(p)), key=lambda i: (-p[i], i))
    out = {
        "top1": int(r == 1),
        "top3": int(r is not None and r <= 3),
        "top5": int(r is not None and r <= 5),
        "rr": 1.0 / r if r else 0.0,
        "conf": p[top],
        "correct": int(top in gold),
        "nll": None,
        "brier": None,
    }
    if gold:
        mass = sum(p[g] for g in gold)
        out["nll"] = -math.log(max(mass, EPS))
        w = 1.0 / len(gold)
        out["brier"] = sum((p[i] - (w if i in gold else 0.0)) ** 2 for i in range(len(p)))
    return out


def ece(confs: Sequence[float], correct: Sequence[int], bins: int = 10) -> float:
    n = len(confs)
    if n == 0:
        return float("nan")
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if (lo < c <= hi) or (b == 0 and c == 0)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            conf = sum(confs[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(acc - conf)
    return total


def reliability(confs: Sequence[float], correct: Sequence[int], bins: int = 10) -> list[dict]:
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if (lo < c <= hi) or (b == 0 and c == 0)]
        out.append({
            "bin": [lo, hi],
            "n": len(idx),
            "accuracy": sum(correct[i] for i in idx) / len(idx) if idx else None,
            "confidence": sum(confs[i] for i in idx) / len(idx) if idx else None,
        })
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(rows: list[dict]) -> dict:
    """rows: per-bug outputs of `score`."""
    return {
        "n": len(rows),
        "n_gold_in_pool": sum(1 for r in rows if r["nll"] is not None),
        "top1": _mean([r["top1"] for r in rows]),
        "top3": _mean([r["top3"] for r in rows]),
        "top5": _mean([r["top5"] for r in rows]),
        "mrr": _mean([r["rr"] for r in rows]),
        "nll": _mean([r["nll"] for r in rows]),
        "brier": _mean([r["brier"] for r in rows]),
        "ece": ece([r["conf"] for r in rows], [r["correct"] for r in rows]),
        "mean_conf": _mean([r["conf"] for r in rows]),
    }


def fit_temperature(dists: list[Sequence[float]], golds: list[Sequence[int]]) -> float:
    """Scalar temperature T minimising gold NLL of softmax(log p / T) (golden-section on log T)."""
    pairs = [(d, g) for d, g in zip(dists, golds) if g]

    def nll(t):
        tot = 0.0
        for d, g in pairs:
            logs = [math.log(max(x, EPS)) / t for x in d]
            m = max(logs)
            z = m + math.log(sum(math.exp(x - m) for x in logs))
            mass = sum(math.exp(logs[i] - z) for i in g)
            tot -= math.log(max(mass, EPS))
        return tot / len(pairs)

    lo, hi = math.log(0.05), math.log(20.0)
    phi = (math.sqrt(5) - 1) / 2
    for _ in range(80):
        a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
        if nll(math.exp(a)) < nll(math.exp(b)):
            hi = b
        else:
            lo = a
    return math.exp((lo + hi) / 2)


def temper(p: Sequence[float], t: float) -> list[float]:
    logs = [math.log(max(x, EPS)) / t for x in p]
    m = max(logs)
    e = [math.exp(x - m) for x in logs]
    z = sum(e)
    return [x / z for x in e]
