"""Runs agents through rounds, applies their actions, then settles.

One `Engine` is one episode (one bug). Pass the same `Exchange` to successive
episodes to carry capital across bugs; that is how earned capital turns into
future market influence.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from .evidence import EvidenceBoard
from .exchange import Exchange, InsufficientFunds
from .protocol import (
    Agent, MarketView, Observation, OpenLocationMarket, ProposePatch, Reveal, Trade,
    TradeDistribution,
)
from .settlement import MarketForecaster, Patch, SettlementConfig, SettlementResult, settle


def location_market_id(target: str) -> str:
    return f"loc:{target}"


@dataclass
class EngineConfig:
    rounds: int = 8
    starting_capital: float = 1000.0
    b: float = 100.0
    max_markets: int = 40
    max_fraction_per_trade: float = 0.25
    max_actions_per_turn: int = 16
    shuffle_turn_order: bool = True
    seed_locations: tuple[str, ...] = ()
    settlement: SettlementConfig = field(default_factory=SettlementConfig)
    seed: int = 0


@dataclass
class RunLog:
    events: list[dict] = field(default_factory=list)

    def add(self, **kw) -> None:
        self.events.append(kw)

    def of_type(self, t: str) -> list[dict]:
        return [e for e in self.events if e.get("type") == t]


class Engine:
    def __init__(
        self,
        agents: list[Agent],
        run_tests: Callable[[Patch], bool] | None = None,
        forecaster: MarketForecaster | None = None,
        cfg: EngineConfig | None = None,
        task: dict | None = None,
        exchange: Exchange | None = None,
    ):
        self.cfg = cfg or EngineConfig()
        self.agents = agents
        names = [a.name for a in agents]
        if len(set(names)) != len(names):
            raise ValueError("agent names must be unique")
        self.run_tests = run_tests
        self.forecaster = forecaster
        self.task = task or {}
        self.rng = random.Random(self.cfg.seed)
        self.ex = exchange or Exchange(
            starting_capital=self.cfg.starting_capital,
            default_b=self.cfg.b,
            max_markets=self.cfg.max_markets,
            max_fraction_per_trade=self.cfg.max_fraction_per_trade,
        )
        self.board = EvidenceBoard()
        self.patches: list[Patch] = []
        self.log = RunLog()
        for a in agents:
            self.ex.register(a.name)
        for loc in self.cfg.seed_locations:
            self._open_location(loc)

    # ---------- helpers ----------
    def _open_location(self, target: str) -> str:
        mid = location_market_id(target)
        if mid not in self.ex.markets:
            self.ex.open_market(mid, f"Is the bug in {target}()?", kind="location", target=target)
        return mid

    def observe(self, agent: str, step: int) -> Observation:
        markets = tuple(
            MarketView(
                m.market_id, m.question, tuple(m.outcomes), tuple(m.prices()),
                m.meta.get("kind", ""), m.meta.get("target"),
            )
            for m in self.ex.open_markets()
        )
        positions = {
            mid: tuple(m.positions[agent])
            for mid, m in self.ex.markets.items()
            if agent in m.positions and m.is_open
        }
        first = {r.fact: r for r in self.board.counted()}
        return Observation(
            step=step,
            agent=agent,
            capital=self.ex.accounts[agent].capital,
            positions=positions,
            markets=markets,
            public_facts=tuple((f, r.text, r.agent) for f, r in first.items()),
            patches=tuple((p.patch_id, p.author, tuple(sorted(p.targets))) for p in self.patches),
            task=self.task,
        )

    def apply(self, agent: str, action, step: int) -> None:
        if isinstance(action, Reveal):
            r = self.board.post(agent, action.fact, action.text, step)
            self.log.add(step=step, agent=agent, type="reveal", fact=r.fact, counts=r.counts)
        elif isinstance(action, OpenLocationMarket):
            try:
                mid = self._open_location(action.target)
                self.log.add(step=step, agent=agent, type="open", market=mid)
            except RuntimeError as e:
                self.log.add(step=step, agent=agent, type="error", error=str(e))
        elif isinstance(action, (Trade, TradeDistribution)):
            m = self.ex.markets.get(action.market_id)
            if m is None or not m.is_open:
                self.log.add(step=step, agent=agent, type="error", error=f"no open market {action.market_id}")
                return
            try:
                if isinstance(action, Trade):
                    t = self.ex.trade_to_belief(agent, action.market_id, action.belief, step, action.outcome)
                else:
                    t = self.ex.trade_toward(agent, action.market_id, list(action.belief), step, action.budget)
            except (InsufficientFunds, ValueError) as e:
                self.log.add(step=step, agent=agent, type="error", error=str(e))
                return
            if t:
                self.log.add(
                    step=step, agent=agent, type="trade", market=action.market_id,
                    outcome=t.outcome, cost=round(t.cost, 4),
                    price_after=round(t.price_after, 4),
                )
        elif isinstance(action, ProposePatch):
            p = Patch(
                f"P{len(self.patches)}", agent, step,
                frozenset(action.targets), frozenset(action.cites), action.diff,
            )
            self.patches.append(p)
            try:
                self.ex.open_market(
                    p.market_id, f"Will patch {p.patch_id} pass the tests?",
                    kind="patch", target=p.patch_id,
                )
            except RuntimeError as e:
                self.log.add(step=step, agent=agent, type="error", error=str(e))
            self.log.add(step=step, agent=agent, type="patch", patch=p.patch_id, targets=sorted(p.targets))
        else:
            raise TypeError(f"unknown action {action!r}")

    # ---------- main loop ----------
    def play(self) -> None:
        for step in range(self.cfg.rounds):
            order = list(self.agents)
            if self.cfg.shuffle_turn_order:
                self.rng.shuffle(order)
            for a in order:
                actions = a.act(self.observe(a.name, step))[: self.cfg.max_actions_per_turn]
                for action in actions:
                    self.apply(a.name, action, step)

    def run(self, settle_fn: Callable[["Engine"], object] | None = None):
        """Play all rounds, then settle (default: patch/test settlement)."""
        self.play()
        if settle_fn is not None:
            return settle_fn(self)
        if self.run_tests is None or self.forecaster is None:
            raise ValueError("default settlement needs run_tests and forecaster")
        return settle(
            self.ex, self.board, self.patches, self.run_tests, self.forecaster, self.cfg.settlement
        )
