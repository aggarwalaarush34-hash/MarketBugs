"""Logarithmic market scoring rule (Hanson 2003) market maker.

A market has n mutually exclusive outcomes. The maker tracks outstanding
shares q[i]; the cost function is

    C(q) = b * ln(sum_i exp(q_i / b))

and the price of outcome i is softmax(q / b)_i. Buying a share vector d costs
C(q + d) - C(q). A share of outcome i pays one credit if i resolves true (or
w_i credits under a weighted resolution, used when a bug has several gold
functions).

If the market opens at prices p0 (rather than uniform), the maker's
worst-case loss is b * ln(1 / p0_i) for the outcome i that resolves true,
so the bound is b * ln(1 / min_i p0_i). For uniform opening this is b ln n.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

PRICE_FLOOR = 1e-12


def logsumexp(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def normalize(xs: list[float]) -> list[float]:
    z = sum(xs)
    if z <= 0:
        raise ValueError("cannot normalize a non-positive vector")
    return [x / z for x in xs]


def geometric_mix(p: list[float], q: list[float], alpha: float) -> list[float]:
    """p^(1-alpha) * q^alpha, renormalised (log-linear interpolation)."""
    logs = [
        (1 - alpha) * math.log(max(a, PRICE_FLOOR)) + alpha * math.log(max(b, PRICE_FLOOR))
        for a, b in zip(p, q)
    ]
    m = max(logs)
    return normalize([math.exp(x - m) for x in logs])


class MarketClosed(Exception):
    pass


@dataclass
class Trade:
    agent: str
    shares: list[float]
    cost: float
    prices_before: list[float]
    prices_after: list[float]
    step: int

    @property
    def outcome(self) -> int:
        """Outcome receiving the most shares (the 'side' of a binary trade)."""
        return max(range(len(self.shares)), key=lambda i: self.shares[i])

    @property
    def price_after(self) -> float:
        return self.prices_after[self.outcome]


@dataclass
class LMSRMarket:
    market_id: str
    question: str
    outcomes: list[str] = field(default_factory=lambda: ["NO", "YES"])
    b: float = 100.0
    q: list[float] = field(default_factory=list)
    positions: dict[str, list[float]] = field(default_factory=dict)
    spent: dict[str, float] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    resolved: int | None = None              # outcome index, -1 if voided, -2 if weighted
    resolution: list[float] | None = None    # payout weight per outcome once resolved
    meta: dict = field(default_factory=dict)
    _q0: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if len(self.outcomes) < 2:
            raise ValueError("market needs at least two outcomes")
        if self.b <= 0:
            raise ValueError("liquidity b must be positive")
        if not self.q:
            self.q = [0.0] * len(self.outcomes)
        if len(self.q) != len(self.outcomes):
            raise ValueError("q must have one entry per outcome")
        self._q0 = list(self.q)

    @classmethod
    def opening_at(cls, market_id: str, question: str, prices: list[float], b: float, **kw):
        """Open a market whose initial prices equal `prices`."""
        prices = normalize([max(p, PRICE_FLOOR) for p in prices])
        q = [b * math.log(p) for p in prices]
        lo = min(q)
        q = [x - lo for x in q]
        return cls(market_id, question, [str(i) for i in range(len(prices))], b, q=q, **kw)

    # ---------- pricing ----------
    @property
    def is_open(self) -> bool:
        return self.resolved is None

    def cost(self, q: list[float] | None = None) -> float:
        q = self.q if q is None else q
        return self.b * logsumexp([x / self.b for x in q])

    def prices(self) -> list[float]:
        z = logsumexp([x / self.b for x in self.q])
        return [math.exp(x / self.b - z) for x in self.q]

    def price(self, outcome: int = 1) -> float:
        return self.prices()[outcome]

    def opening_prices(self) -> list[float]:
        z = logsumexp([x / self.b for x in self._q0])
        return [math.exp(x / self.b - z) for x in self._q0]

    def quote(self, outcome: int, shares: float) -> float:
        d = [0.0] * len(self.q)
        d[outcome] = shares
        return self.quote_vector(d)

    def quote_vector(self, d: list[float]) -> float:
        return self.cost([a + b for a, b in zip(self.q, d)]) - self.cost()

    def shares_to_reach(self, outcome: int, target: float) -> float:
        """Shares of `outcome` to buy so its price becomes `target` (others fixed)."""
        if not 0.0 < target < 1.0:
            raise ValueError("target price must be in (0, 1)")
        others = [x / self.b for j, x in enumerate(self.q) if j != outcome]
        new_qi = self.b * (math.log(target / (1 - target)) + logsumexp(others))
        return new_qi - self.q[outcome]

    def shares_to_prices(self, target: list[float]) -> list[float]:
        """Buy-only share vector that moves prices exactly to `target`.

        q + d must equal b*log(target) + c for some constant c; choosing c so
        that min(d) = 0 gives the cheapest buy-only vector.
        """
        target = normalize([max(t, PRICE_FLOOR) for t in target])
        d = [self.b * math.log(t) - qi for t, qi in zip(target, self.q)]
        lo = min(d)
        return [x - lo for x in d]

    # ---------- trading ----------
    def buy_vector(self, agent: str, d: list[float], step: int = 0) -> Trade:
        if not self.is_open:
            raise MarketClosed(self.market_id)
        if len(d) != len(self.q):
            raise ValueError("share vector has wrong length")
        before = self.prices()
        c = self.quote_vector(d)
        self.q = [a + b for a, b in zip(self.q, d)]
        pos = self.positions.setdefault(agent, [0.0] * len(self.outcomes))
        for i, x in enumerate(d):
            pos[i] += x
        self.spent[agent] = self.spent.get(agent, 0.0) + c
        t = Trade(agent, list(d), c, before, self.prices(), step)
        self.trades.append(t)
        return t

    def buy(self, agent: str, outcome: int, shares: float, step: int = 0) -> Trade:
        d = [0.0] * len(self.q)
        d[outcome] = shares
        return self.buy_vector(agent, d, step)

    # ---------- settlement ----------
    def resolve(self, outcome: int) -> dict[str, float]:
        w = [0.0] * len(self.outcomes)
        w[outcome] = 1.0
        payouts = self.resolve_weighted(w)
        self.resolved = outcome
        return payouts

    def resolve_weighted(self, weights: list[float]) -> dict[str, float]:
        """Close with payout weight per outcome (weights sum to 1)."""
        if not self.is_open:
            raise MarketClosed(self.market_id)
        if len(weights) != len(self.outcomes) or abs(sum(weights) - 1) > 1e-9 or min(weights) < 0:
            raise ValueError("weights must be a distribution over outcomes")
        self.resolution = list(weights)
        self.resolved = -2
        return {a: sum(w * x for w, x in zip(weights, pos)) for a, pos in self.positions.items()}

    def void(self) -> dict[str, float]:
        """Cancel the market; returns each trader's refund."""
        if not self.is_open:
            raise MarketClosed(self.market_id)
        self.resolved = -1
        self.resolution = None
        self.meta["voided"] = True
        return dict(self.spent)

    def payout(self, agent: str) -> float:
        if self.resolution is None:
            return self.spent.get(agent, 0.0) if self.resolved == -1 else 0.0
        pos = self.positions.get(agent, [0.0] * len(self.outcomes))
        return sum(w * x for w, x in zip(self.resolution, pos))

    def pnl(self) -> dict[str, float]:
        if self.is_open:
            raise ValueError("market not resolved")
        if self.resolved == -1:
            return {a: 0.0 for a in self.positions}
        return {a: self.payout(a) - self.spent.get(a, 0.0) for a in self.positions}

    def maker_loss(self) -> float:
        """Subsidy actually paid: payouts minus credits collected (negative = maker profit)."""
        if self.is_open:
            raise ValueError("market not resolved")
        if self.resolved == -1:
            return 0.0
        collected = self.cost() - self.cost(self._q0)
        paid = sum(w * (qf - q0) for w, qf, q0 in zip(self.resolution, self.q, self._q0))
        return paid - collected

    @property
    def max_maker_loss(self) -> float:
        return self.b * math.log(1.0 / min(self.opening_prices()))
