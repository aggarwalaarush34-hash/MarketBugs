"""Non-market aggregation baselines over the same cached Gemma forecasts.

All baselines read the same five single-note forecasts (or the all-notes
forecast) the market agents use, so comparisons are matched on inputs.
Anything with fitted parameters (temperature, Hedge weights) is fitted on
the development split only.
"""

from __future__ import annotations

import math
from typing import Sequence

from ..analysis.metrics import fit_temperature, temper
from ..data import AGENTS, FULL, Bug, bit
from ..lmsr import normalize

EPS = 1e-12


def logpool(dists: Sequence[Sequence[float]], weights: Sequence[float] | None = None) -> list[float]:
    weights = list(weights) if weights is not None else [1.0 / len(dists)] * len(dists)
    k = len(dists[0])
    logs = [sum(w * math.log(max(d[i], EPS)) for w, d in zip(weights, dists)) for i in range(k)]
    m = max(logs)
    return normalize([math.exp(x - m) for x in logs])


def linpool(dists: Sequence[Sequence[float]], weights: Sequence[float] | None = None) -> list[float]:
    weights = list(weights) if weights is not None else [1.0 / len(dists)] * len(dists)
    k = len(dists[0])
    return normalize([sum(w * d[i] for w, d in zip(weights, dists)) for i in range(k)])


def singles(bug: Bug) -> list[tuple[float, ...]]:
    return [bug.forecast(bit(a)) for a in AGENTS]


def hedge_weights(dev: list[Bug], eta: float) -> list[float]:
    """Multiplicative weights on per-specialist gold log loss over the dev sequence."""
    w = [1.0] * len(AGENTS)
    for b in dev:
        if not b.gold_in_pool:
            continue
        for i, a in enumerate(AGENTS):
            loss = -math.log(max(b.gold_mass(bit(a)), EPS))
            w[i] *= math.exp(-eta * loss)
        z = sum(w)
        w = [x / z for x in w]
    return w


def choose_hedge(dev: list[Bug], etas=(0.1, 0.3, 1.0, 3.0)) -> tuple[float, list[float]]:
    """Pick eta by 2-fold CV on dev (even/odd bugs), then refit on all of dev."""
    def cv(eta):
        tot, n = 0.0, 0
        for train, test in ((dev[0::2], dev[1::2]), (dev[1::2], dev[0::2])):
            w = hedge_weights(train, eta)
            for b in test:
                if b.gold_in_pool:
                    p = logpool(singles(b), w)
                    tot -= math.log(max(sum(p[g] for g in b.gold), EPS))
                    n += 1
        return tot / max(n, 1)

    best = min(etas, key=lambda e: (cv(e), e))
    return best, hedge_weights(dev, best)


class Baselines:
    """Fits dev-only parameters once, then produces every baseline's forecast for a bug."""

    def __init__(self, dev: list[Bug]):
        self.t_all = fit_temperature([b.forecast(FULL) for b in dev], [b.gold for b in dev])
        self.t_logpool = fit_temperature([logpool(singles(b)) for b in dev], [b.gold for b in dev])
        self.hedge_eta, self.hedge_w = choose_hedge(dev)
        self.t_hedge = fit_temperature([logpool(singles(b), self.hedge_w) for b in dev], [b.gold for b in dev])

    def params(self) -> dict:
        return {
            "temperature_all_notes": self.t_all,
            "temperature_logpool": self.t_logpool,
            "hedge_eta": self.hedge_eta,
            "hedge_weights": dict(zip(AGENTS, self.hedge_w)),
            "temperature_hedge": self.t_hedge,
        }

    def forecasts(self, bug: Bug) -> dict[str, list[float]]:
        s = singles(bug)
        out = {"no_notes": list(bug.forecast(0))}
        for a, d in zip(AGENTS, s):
            out[f"single.{a}"] = list(d)
        out["all_notes"] = list(bug.forecast(FULL))
        out["all_notes.T"] = temper(bug.forecast(FULL), self.t_all)
        out["logpool"] = logpool(s)
        out["logpool.T"] = temper(out["logpool"], self.t_logpool)
        out["linpool"] = linpool(s)
        out["hedge"] = logpool(s, self.hedge_w)
        out["hedge.T"] = temper(out["hedge"], self.t_hedge)
        return out
