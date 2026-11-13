"""Paired bootstrap confidence intervals.

The primary interval is a two-stage repository-cluster bootstrap: resample
repositories with replacement, then bugs within each sampled repository with
replacement. With seven held-out repositories this interval is wide; we
report the instance-level interval next to it, never instead of it.
"""

from __future__ import annotations

import random
from typing import Sequence


def _percentile(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        return float("nan")
    i = min(len(sorted_xs) - 1, max(0, int(round(q * (len(sorted_xs) - 1)))))
    return sorted_xs[i]


def cluster_bootstrap(
    diffs: Sequence[float], clusters: Sequence[str], reps: int = 10000, seed: int = 20261112
) -> dict:
    if len(diffs) != len(clusters):
        raise ValueError("diffs and clusters differ in length")
    rng = random.Random(seed)
    by: dict[str, list[float]] = {}
    for d, c in zip(diffs, clusters):
        by.setdefault(c, []).append(d)
    keys = sorted(by)
    stats = []
    for _ in range(reps):
        tot, n = 0.0, 0
        for _ in keys:
            grp = by[rng.choice(keys)]
            for _ in grp:
                tot += rng.choice(grp)
                n += 1
        stats.append(tot / n)
    stats.sort()
    return {
        "mean": sum(diffs) / len(diffs),
        "lo": _percentile(stats, 0.025),
        "hi": _percentile(stats, 0.975),
        "reps": reps,
        "seed": seed,
        "clusters": len(keys),
        "kind": "two-stage repository cluster",
    }


def instance_bootstrap(diffs: Sequence[float], reps: int = 10000, seed: int = 20261112) -> dict:
    rng = random.Random(seed)
    n = len(diffs)
    stats = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(reps))
    return {
        "mean": sum(diffs) / n,
        "lo": _percentile(stats, 0.025),
        "hi": _percentile(stats, 0.975),
        "reps": reps,
        "seed": seed,
        "kind": "instance",
    }


def paired(a: Sequence[float], b: Sequence[float], clusters: Sequence[str], reps: int = 10000, seed: int = 20261112) -> dict:
    """CIs for mean(a - b), skipping pairs where either side is None."""
    keep = [(x, y, c) for x, y, c in zip(a, b, clusters) if x is not None and y is not None]
    diffs = [x - y for x, y, _ in keep]
    cl = [c for _, _, c in keep]
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    return {
        "n": len(diffs),
        "wins": wins,
        "losses": losses,
        "ties": len(diffs) - wins - losses,
        "cluster": cluster_bootstrap(diffs, cl, reps, seed),
        "instance": instance_bootstrap(diffs, reps, seed),
    }
