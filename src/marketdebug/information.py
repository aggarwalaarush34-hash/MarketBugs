"""Information rewards: marginal, Shapley and pivotal credit for evidence.

All three need a *forecaster*: a function from a set of facts to the
probability that a target hypothesis is true. The coalition value is

    v(S) = s(forecast(facts of S), x) - s(forecast({}), x)

where s is a proper scoring rule and x the realised outcome (0 or 1).
These are Conitzer-inspired mechanisms; their incentive guarantees do not
automatically carry over to LLM agents and should be tested, not assumed.
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Callable, Iterable, Sequence

from .evidence import Report

Forecaster = Callable[[frozenset[str]], float]
EPS = 1e-6


def quadratic_score(p: float, x: int) -> float:
    return 1.0 - (p - x) ** 2


def log_score(p: float, x: int) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p if x == 1 else 1 - p)


RULES = {"quadratic": quadratic_score, "log": log_score}


class CoalitionValue:
    """Caches forecasts so Shapley sampling doesn't re-query an expensive judge."""

    def __init__(self, forecaster: Forecaster, x: int, rule: str = "quadratic"):
        self.forecaster = forecaster
        self.x = x
        self.score = RULES[rule]
        self._cache: dict[frozenset[str], float] = {}
        self.base = self.score(self.forecast(frozenset()), x)

    def forecast(self, facts: frozenset[str]) -> float:
        if facts not in self._cache:
            self._cache[facts] = self.forecaster(facts)
        return self._cache[facts]

    def __call__(self, facts: Iterable[str]) -> float:
        return self.score(self.forecast(frozenset(facts)), self.x) - self.base

    @property
    def calls(self) -> int:
        return len(self._cache)


def _facts_of(players: dict[str, list[Report]], members: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for m in members:
        out.update(r.fact for r in players[m])
    return out


def marginal_rewards(reports: Sequence[Report], value: CoalitionValue) -> dict[str, float]:
    """Pay each counted report its score change, in the order it was filed."""
    rewards: dict[str, float] = {}
    seen: set[str] = set()
    prev = value(seen)
    for r in sorted((r for r in reports if r.counts), key=lambda r: (r.step, r.report_id)):
        seen.add(r.fact)
        cur = value(seen)
        rewards[r.agent] = rewards.get(r.agent, 0.0) + (cur - prev)
        prev = cur
    return rewards


def shapley_rewards(
    players: dict[str, list[Report]],
    value: CoalitionValue,
    samples: int = 2000,
    exact_up_to: int = 8,
    rng: random.Random | None = None,
) -> dict[str, float]:
    """Shapley value over agents. Exact for small n, permutation-sampled otherwise."""
    names = sorted(players)
    n = len(names)
    if n == 0:
        return {}
    phi = {a: 0.0 for a in names}
    if n <= exact_up_to:
        perms: Iterable[Sequence[str]] = itertools.permutations(names)
        count = math.factorial(n)
    else:
        rng = rng or random.Random(0)
        def gen():
            for _ in range(samples):
                p = names[:]
                rng.shuffle(p)
                yield p
        perms, count = gen(), samples
    for perm in perms:
        facts: set[str] = set()
        prev = value(facts)
        for a in perm:
            facts |= {r.fact for r in players[a]}
            cur = value(facts)
            phi[a] += cur - prev
            prev = cur
    return {a: v / count for a, v in phi.items()}


def pivotal_rewards(players: dict[str, list[Report]], value: CoalitionValue) -> dict[str, float]:
    """v(everyone) - v(everyone but i). Not budget-balanced in general."""
    names = list(players)
    full = value(_facts_of(players, names))
    return {a: full - value(_facts_of(players, [b for b in names if b != a])) for a in names}


def shapley_additive_approx(players: dict[str, list[Report]], value: CoalitionValue) -> dict[str, float]:
    """Shapley value assuming non-additive interaction only in the grand coalition.

    phi_i = v({i}) + (v(N) - sum_j v({j})) / n. Exact when v(S) = sum_{i in S} v({i})
    for every |S| < n. Needs only n + 1 forecasts instead of 2^n, which is what
    is available when only singleton and full coalitions were evaluated.
    """
    names = sorted(players)
    if not names:
        return {}
    single = {a: value(_facts_of(players, [a])) for a in names}
    resid = (value(_facts_of(players, names)) - sum(single.values())) / len(names)
    return {a: s + resid for a, s in single.items()}


def marginal_by_order(players: dict[str, list[Report]], value: CoalitionValue) -> dict[str, list[float]]:
    """Each agent's marginal credit under every reveal order (n! orders, exact)."""
    names = sorted(players)
    out: dict[str, list[float]] = {a: [] for a in names}
    for perm in itertools.permutations(names):
        facts: set[str] = set()
        prev = value(facts)
        for a in perm:
            facts |= {r.fact for r in players[a]}
            cur = value(facts)
            out[a].append(cur - prev)
            prev = cur
    return out


MECHANISMS = ("marginal", "shapley", "shapley_approx", "pivotal")


def information_rewards(
    mechanism: str,
    reports: Sequence[Report],
    players: dict[str, list[Report]],
    value: CoalitionValue,
    scale: float = 100.0,
    **kw,
) -> dict[str, float]:
    if mechanism == "marginal":
        raw = marginal_rewards(reports, value)
    elif mechanism == "shapley":
        raw = shapley_rewards(players, value, **kw)
    elif mechanism == "shapley_approx":
        raw = shapley_additive_approx(players, value)
    elif mechanism == "pivotal":
        raw = pivotal_rewards(players, value)
    elif mechanism == "none":
        raw = {}
    else:
        raise ValueError(mechanism)
    return {a: r * scale for a, r in raw.items()}
