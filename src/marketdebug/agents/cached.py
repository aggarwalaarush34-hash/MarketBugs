"""Agents whose beliefs are cached Gemma 4 E4B forecasts.

A specialist owns one note (its view of the bug). Its belief is the frozen
Gemma probability agent's forecast given its own note and, when
`condition_on_public` is set and the forecast exists, every note already on
the public board. Beliefs are therefore real Gemma outputs; the agent's
choices are only when to reveal and how much to trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..data import Bug, MissingCoalition, bit
from ..protocol import Observation, Reveal, TradeDistribution

REVEAL_POLICIES = ("trade_then_reveal", "reveal_first", "hoard")


def note_fact(specialist: str) -> str:
    return f"note:{specialist}"


def mask_of_facts(facts) -> int:
    m = 0
    for f in facts:
        if f.startswith("note:"):
            m |= bit(f.split(":", 1)[1])
    return m


def localization_market(obs: Observation):
    for m in obs.markets:
        if m.kind == "localization":
            return m
    return None


@dataclass
class CachedGemmaAgent:
    name: str
    specialist: str
    bug: Bug
    policy: str = "trade_then_reveal"
    per_turn_fraction: float = 0.2 / 3
    condition_on_public: bool = False
    beliefs_used: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.policy not in REVEAL_POLICIES:
            raise ValueError(self.policy)

    def belief_mask(self, obs: Observation) -> int:
        own = bit(self.specialist)
        if not self.condition_on_public:
            return own
        public = mask_of_facts(f for f, _, _ in obs.public_facts)
        mask = own | public
        try:
            self.bug.forecast(mask)
        except MissingCoalition:
            return own
        return mask

    def act(self, obs: Observation):
        actions = []
        fact = note_fact(self.specialist)
        already = any(f == fact for f, _, _ in obs.public_facts)
        reveal = Reveal(fact, f"{self.specialist} note") if not already and self.policy != "hoard" else None
        if reveal and self.policy == "reveal_first":
            actions.append(reveal)
        m = localization_market(obs)
        if m is not None:
            mask = self.belief_mask(obs)
            self.beliefs_used.append(mask)
            actions.append(
                TradeDistribution(m.market_id, self.bug.forecast(mask), budget=obs.capital * self.per_turn_fraction)
            )
        if reveal and self.policy == "trade_then_reveal":
            actions.append(reveal)
        return actions


@dataclass
class CopycatAgent:
    """Re-posts another specialist's note once it is public and trades on that belief.

    Used to test that the evidence board denies copied facts any information
    credit while still letting the copycat trade.
    """

    name: str
    copies: str
    bug: Bug
    per_turn_fraction: float = 0.2 / 3

    def act(self, obs: Observation):
        fact = note_fact(self.copies)
        public = {f for f, _, _ in obs.public_facts}
        if fact not in public:
            return []
        actions = [Reveal(fact, f"copy of {self.copies} note")]
        m = localization_market(obs)
        if m is not None:
            actions.append(
                TradeDistribution(m.market_id, self.bug.forecast(bit(self.copies)), budget=obs.capital * self.per_turn_fraction)
            )
        return actions
