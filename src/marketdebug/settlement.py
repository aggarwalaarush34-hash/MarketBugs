"""Patches, test-ordered settlement, and the repair ledger.

Settlement order:
  1. Test patches in descending patch-market price, up to `test_budget`.
     The first one that passes is the accepted fix.
  2. Resolve patch markets from test results; void any left untested.
  3. Resolve root-cause markets from the accepted patch (a location market
     resolves YES if the accepted patch touches that function). Void them
     if nothing passed.
  4. Book information rewards for each resolved root-cause market.
  5. Book the repair bounty: the author's share plus a share split among
     the original finders of the facts the patch cites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .evidence import EvidenceBoard
from .exchange import Exchange
from .information import CoalitionValue, information_rewards


@dataclass
class Patch:
    patch_id: str
    author: str
    step: int
    targets: frozenset[str]          # functions/files the diff touches
    cites: frozenset[str] = frozenset()  # fact keys from the evidence board
    diff: str = ""

    @property
    def market_id(self) -> str:
        return f"patch:{self.patch_id}"


@dataclass
class SettlementConfig:
    test_budget: int = 5
    info_mechanism: str = "shapley"   # marginal | shapley | pivotal | none
    scoring_rule: str = "quadratic"
    info_scale: float = 100.0
    repair_bounty: float = 200.0
    cite_share: float = 0.3           # fraction of bounty split among cited finders
    shapley_samples: int = 2000


@dataclass
class SettlementResult:
    accepted: Patch | None
    tested: list[tuple[str, bool]] = field(default_factory=list)
    resolutions: dict[str, str] = field(default_factory=dict)
    information: dict[str, float] = field(default_factory=dict)
    repair: dict[str, float] = field(default_factory=dict)
    forecaster_calls: int = 0


# forecaster(market_id, facts) -> P(YES)
MarketForecaster = Callable[[str, frozenset[str]], float]


def settle(
    ex: Exchange,
    board: EvidenceBoard,
    patches: list[Patch],
    run_tests: Callable[[Patch], bool],
    forecaster: MarketForecaster,
    cfg: SettlementConfig | None = None,
    order: str = "price",
) -> SettlementResult:
    cfg = cfg or SettlementConfig()
    res = SettlementResult(accepted=None)

    # 1-2. test in price order (or the given order, for ablations); ties keep proposal order
    if order == "price":
        queue = sorted(
            patches,
            key=lambda p: -ex.markets[p.market_id].price(1) if p.market_id in ex.markets else 0.0,
        )
    elif order == "given":
        queue = list(patches)
    else:
        raise ValueError(order)
    for p in queue:
        mid = p.market_id
        if len(res.tested) < cfg.test_budget and res.accepted is None:
            ok = bool(run_tests(p))
            res.tested.append((p.patch_id, ok))
            if mid in ex.markets:
                ex.resolve(mid, 1 if ok else 0)
                res.resolutions[mid] = "YES" if ok else "NO"
            if ok:
                res.accepted = p
        elif mid in ex.markets and ex.markets[mid].resolved is None:
            ex.void(mid)
            res.resolutions[mid] = "VOID"

    # 3. root-cause markets
    root_markets = [m for m in ex.open_markets() if m.meta.get("kind") == "location"]
    for m in root_markets:
        if res.accepted is None:
            ex.void(m.market_id)
            res.resolutions[m.market_id] = "VOID"
            continue
        yes = m.meta["target"] in res.accepted.targets
        ex.resolve(m.market_id, 1 if yes else 0)
        res.resolutions[m.market_id] = "YES" if yes else "NO"

        # 4. information rewards for this market
        value = CoalitionValue(
            lambda facts, mid=m.market_id: forecaster(mid, facts),
            x=1 if yes else 0,
            rule=cfg.scoring_rule,
        )
        rewards = information_rewards(
            cfg.info_mechanism,
            board.reports,
            board.by_agent(),
            value,
            scale=cfg.info_scale,
            **({"samples": cfg.shapley_samples} if cfg.info_mechanism == "shapley" else {}),
        )
        for a, r in rewards.items():
            res.information[a] = res.information.get(a, 0.0) + r
        res.forecaster_calls += value.calls

    # any other markets still open have no resolver: refund them
    for m in ex.open_markets():
        ex.void(m.market_id)
        res.resolutions[m.market_id] = "VOID"

    ex.book("information", res.information)

    # 5. repair bounty
    if res.accepted is not None:
        p = res.accepted
        finders = {}
        for r in board.counted():
            if r.fact in p.cites:
                finders[r.fact] = r.agent
        cite_pool = cfg.repair_bounty * cfg.cite_share if finders else 0.0
        res.repair[p.author] = cfg.repair_bounty - cite_pool
        for agent in finders.values():
            res.repair[agent] = res.repair.get(agent, 0.0) + cite_pool / len(finders)
        ex.book("repair", res.repair)

    return res
