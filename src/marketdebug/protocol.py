"""What an agent sees and what it can do.

Simulated agents and LLM agents implement the same `Agent.act` interface,
so the engine, markets and reward code never change when you swap them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Union


@dataclass(frozen=True)
class MarketView:
    market_id: str
    question: str
    outcomes: tuple[str, ...]
    prices: tuple[float, ...]
    kind: str            # "location" | "patch" | other
    target: str | None   # function name for location markets, patch id for patch markets


@dataclass(frozen=True)
class Observation:
    step: int
    agent: str
    capital: float
    positions: dict[str, tuple[float, ...]]
    markets: tuple[MarketView, ...]
    public_facts: tuple[tuple[str, str, str], ...]   # (fact, text, finder)
    patches: tuple[tuple[str, str, tuple[str, ...]], ...]  # (patch_id, author, targets)
    task: dict = field(default_factory=dict)          # task description, repo info, etc.


@dataclass(frozen=True)
class Reveal:
    fact: str
    text: str = ""


@dataclass(frozen=True)
class Trade:
    market_id: str
    belief: float          # agent's P(YES); the exchange trades toward it
    outcome: int = 1


@dataclass(frozen=True)
class TradeDistribution:
    """Multi-outcome trade: move the market's prices toward `belief`."""
    market_id: str
    belief: tuple[float, ...]
    budget: float | None = None   # credits; None = exchange default fraction of capital


@dataclass(frozen=True)
class OpenLocationMarket:
    target: str            # function or file the agent suspects


@dataclass(frozen=True)
class ProposePatch:
    targets: tuple[str, ...]
    cites: tuple[str, ...] = ()
    diff: str = ""


Action = Union[Reveal, Trade, TradeDistribution, OpenLocationMarket, ProposePatch]


class Agent(Protocol):
    name: str

    def act(self, obs: Observation) -> list[Action]: ...
