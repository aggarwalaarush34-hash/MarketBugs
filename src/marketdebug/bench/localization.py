"""Fault-localization benchmark: one LMSR market per bug, Gemma specialists as traders.

Per bug (one episode):
  1. Open a 20-outcome LMSR market over the retrieved candidates, at the
     no-notes Gemma forecast (or uniform).
  2. Agents play `rounds` rounds through the Engine: each trades toward its
     belief within a per-turn budget and reveals its note on the board.
  3. Settle: resolve the market with payout weight uniform over the gold
     candidates (void it if retrieval missed the gold function), book
     information rewards for counted notes, and check the books balance.

Capital lives in one Exchange shared across the bugs of a run, so capital
earned on earlier bugs is market influence on later ones.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

from ..agents.cached import CachedGemmaAgent, CopycatAgent, mask_of_facts
from ..data import AGENTS, Bug
from ..engine import Engine, EngineConfig
from ..exchange import Exchange
from ..information import CoalitionValue, information_rewards
from ..lmsr import LMSRMarket


@dataclass(frozen=True)
class MarketConfig:
    b: float = 50.0
    rounds: int = 3
    per_turn_fraction: float = 0.2 / 3
    starting_capital: float = 1000.0
    capital_floor: float = 1.0
    capital_from: tuple[str, ...] = ("trading",)
    info_mechanism: str = "auto"       # auto | shapley | shapley_approx | marginal | pivotal | none
    scoring_rule: str = "log"
    info_scale: float = 100.0          # credits per unit of score
    reveal_policy: str = "trade_then_reveal"
    condition_on_public: bool = False
    open_at: str = "no_notes"          # no_notes | uniform
    copycat_of: str | None = None      # add a copycat agent duplicating this specialist
    seed: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["capital_from"] = list(self.capital_from)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MarketConfig":
        d = dict(d)
        d["capital_from"] = tuple(d.get("capital_from", ("trading",)))
        return cls(**d)


@dataclass
class EpisodeResult:
    bug_id: str
    repo: str
    stratum: str | None
    gold: tuple[int, ...]
    prices: list[float]
    trading_pnl: dict[str, float]
    information: dict[str, float]
    capital_after: dict[str, float]
    counted_reveals: list[str]
    denied_reveals: list[tuple[str, str]]
    voided: bool
    maker_loss: float
    info_mechanism: str
    trades: int = 0
    errors: list[str] = field(default_factory=list)


def make_agents(bug: Bug, cfg: MarketConfig):
    agents = [
        CachedGemmaAgent(
            a, a, bug, cfg.reveal_policy, cfg.per_turn_fraction, cfg.condition_on_public
        )
        for a in AGENTS
    ]
    if cfg.copycat_of:
        agents.append(CopycatAgent(f"copycat_{cfg.copycat_of}", cfg.copycat_of, bug, cfg.per_turn_fraction))
    return agents


def new_exchange(cfg: MarketConfig) -> Exchange:
    return Exchange(
        starting_capital=cfg.starting_capital,
        default_b=cfg.b,
        capital_floor=cfg.capital_floor,
        capital_from=cfg.capital_from,
    )


def gold_value(bug: Bug, rule: str) -> CoalitionValue:
    """v(S) = score of the gold mass given the notes in S, minus the no-notes score."""
    return CoalitionValue(lambda facts: bug.gold_mass(mask_of_facts(facts)), x=1, rule=rule)


def run_episode(bug: Bug, ex: Exchange, cfg: MarketConfig, rng: random.Random) -> EpisodeResult:
    open_p = bug.forecast(0) if cfg.open_at == "no_notes" else [1.0 / bug.k] * bug.k
    mid = f"bug:{bug.id}"
    m = LMSRMarket.opening_at(mid, f"Which candidate contains the bug in {bug.id}?", list(open_p), cfg.b)
    m.meta.update(kind="localization", target=bug.id)
    ex.add_market(m)

    agents = make_agents(bug, cfg)
    eng = Engine(
        agents,
        cfg=EngineConfig(rounds=cfg.rounds, seed=rng.randrange(1 << 30)),
        exchange=ex,
        task={"bug": bug.id},
    )
    eng.play()

    voided = not bug.gold_in_pool
    prices = m.prices()
    if voided:
        ex.void(mid)
        pnl = {a.name: 0.0 for a in agents}
    else:
        ex.resolve_weighted(mid, bug.resolution_weights())
        pnl = {a.name: m.pnl().get(a.name, 0.0) for a in agents}

    info: dict[str, float] = {}
    mech = cfg.info_mechanism
    if mech == "auto":
        mech = "shapley" if bug.has_full_table else "shapley_approx"
    if not voided and mech != "none":
        players = eng.board.by_agent()
        if players:
            info = information_rewards(
                mech, eng.board.reports, players, gold_value(bug, cfg.scoring_rule), scale=cfg.info_scale
            )
            ex.book("information", info, note=bug.id)

    return EpisodeResult(
        bug_id=bug.id,
        repo=bug.repo,
        stratum=bug.stratum,
        gold=bug.gold,
        prices=prices,
        trading_pnl=pnl,
        information=info,
        capital_after={a: acct.capital for a, acct in ex.accounts.items()},
        counted_reveals=[r.agent for r in eng.board.counted()],
        denied_reveals=[(r.agent, r.fact) for r in eng.board.reports if not r.counts],
        voided=voided,
        maker_loss=m.maker_loss(),
        info_mechanism=mech if not voided else "none",
        trades=len(m.trades),
        errors=[e["error"] for e in eng.log.of_type("error")],
    )


def run_split(
    bugs: list[Bug], cfg: MarketConfig, ex: Exchange | None = None, order_seed: int | None = None
) -> tuple[list[EpisodeResult], Exchange]:
    """Run every bug in sequence on one exchange (capital carries across bugs)."""
    ex = ex or new_exchange(cfg)
    rng = random.Random(cfg.seed)
    seq = list(bugs)
    if order_seed is not None:
        random.Random(order_seed).shuffle(seq)
    results = [run_episode(b, ex, cfg, rng) for b in seq]
    audit = ex.audit()
    if not audit["balanced"]:
        raise RuntimeError(f"books do not balance: {audit['problems']}")
    return results, ex
