"""SYNTHETIC simulation of the repair ledger and patch markets.

No Gemma-generated patches exist for MarketBugs, so the repair side of the
mechanism is exercised on a synthetic world. Results from this module are
about the mechanism's logic (does price-ordered testing find the fix with
fewer test runs; does the bounty reach the right agents), not about Gemma.

World: one bug in one of `n_locations` functions. Each agent receives noisy
private evidence (a fact pointing at a location, correct with probability
`skill`), forms a posterior, trades in location markets, and, if its top
location is confident enough, proposes a patch. A patch passes iff it
targets the true location and the agent's patch-writing draw succeeds
(probability `patch_quality`). Agents trade on patch markets using
P(pass) = P(location correct | own posterior) * patch_quality.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field

from ..engine import Engine, EngineConfig
from ..protocol import Observation, OpenLocationMarket, ProposePatch, Reveal, Trade
from ..settlement import Patch, SettlementConfig


@dataclass
class SimAgent:
    name: str
    signal: str            # location its private evidence points to
    skill: float           # P(signal is the true location)
    locations: list[str]
    patch_quality: float
    propose_threshold: float = 0.45
    proposed: bool = False

    def posterior(self, public: list[tuple[str, float]]) -> dict[str, float]:
        """Naive-Bayes posterior over locations from own signal plus public signals."""
        n = len(self.locations)
        logp = {l: 0.0 for l in self.locations}
        for sig, sk in [(self.signal, self.skill), *public]:
            for l in self.locations:
                logp[l] += math.log(sk if l == sig else (1 - sk) / (n - 1))
        m = max(logp.values())
        z = sum(math.exp(v - m) for v in logp.values())
        return {l: math.exp(v - m) / z for l, v in logp.items()}

    def act(self, obs: Observation):
        public = [(f.split(":")[1], float(f.split(":")[2])) for f, _, who in obs.public_facts if who != self.name]
        post = self.posterior(public)
        actions = [OpenLocationMarket(self.signal), Reveal(f"sig:{self.signal}:{self.skill:.3f}:{self.name}")]
        for m in obs.markets:
            if m.kind == "location":
                actions.append(Trade(m.market_id, post[m.target]))
            elif m.kind == "patch":
                targets = next((t for pid, _, t in obs.patches if pid == m.target), ())
                p_loc = sum(post[t] for t in targets)
                actions.append(Trade(m.market_id, p_loc * self.patch_quality))
        top = max(post, key=post.get)
        if not self.proposed and post[top] >= self.propose_threshold:
            self.proposed = True
            cites = tuple(f for f, _, _ in obs.public_facts if f.split(":")[1] == top)
            actions.append(ProposePatch((top,), cites))
        return actions


@dataclass
class SimConfig:
    n_agents: int = 8
    n_locations: int = 6
    skill_range: tuple[float, float] = (0.35, 0.8)
    patch_quality: float = 0.7
    rounds: int = 4
    test_budget: int = 3
    b: float = 50.0


@dataclass
class SimOutcome:
    solved: bool
    tests_used: int
    n_patches: int
    oracle_any_passing: bool
    repair: dict[str, float]
    author_correct: bool
    order: str
    extras: dict = field(default_factory=dict)


def _world(cfg: SimConfig, rng: random.Random):
    locs = [f"f{i}" for i in range(cfg.n_locations)]
    true = rng.choice(locs)
    agents = []
    for i in range(cfg.n_agents):
        skill = rng.uniform(*cfg.skill_range)
        sig = true if rng.random() < skill else rng.choice([l for l in locs if l != true])
        agents.append(SimAgent(f"a{i}", sig, skill, locs, cfg.patch_quality))
    quality_draws = {a.name: rng.random() < cfg.patch_quality for a in agents}
    return locs, true, agents, quality_draws


def run_once(cfg: SimConfig, seed: int, order: str = "price") -> SimOutcome:
    rng = random.Random(seed)
    locs, true, agents, quality = _world(cfg, rng)

    def run_tests(p: Patch) -> bool:
        return true in p.targets and quality[p.author]

    def forecaster(mid: str, facts: frozenset[str]) -> float:
        target = mid.split(":", 1)[1]
        public = [(f.split(":")[1], float(f.split(":")[2])) for f in facts if f.startswith("sig:")]
        probe = SimAgent("probe", target, 1.0 / len(locs), locs, 0.0)
        return probe.posterior(public)[target] if public else 1.0 / len(locs)

    eng = Engine(
        agents, run_tests, forecaster,
        EngineConfig(rounds=cfg.rounds, b=cfg.b, seed=seed,
                     settlement=SettlementConfig(test_budget=cfg.test_budget, info_mechanism="shapley",
                                                 shapley_samples=300, scoring_rule="log")),
    )
    eng.play()
    patches = list(eng.patches)
    if order == "random":
        rng.shuffle(patches)
    from ..settlement import settle

    res = settle(eng.ex, eng.board, patches, run_tests, forecaster, eng.cfg.settlement,
                 order="price" if order == "price" else "given")
    oracle = any(run_tests(p) for p in eng.patches)
    true_sig = {a.name for a in agents if a.signal == true}
    info = res.information
    t_vals = [info.get(a.name, 0.0) for a in agents if a.name in true_sig]
    f_vals = [info.get(a.name, 0.0) for a in agents if a.name not in true_sig]
    extras = {
        "info_true_signal_mean": statistics.mean(t_vals) if t_vals else None,
        "info_false_signal_mean": statistics.mean(f_vals) if f_vals else None,
        "repair_to_true_signal": sum(v for a, v in res.repair.items() if a in true_sig),
        "repair_total": sum(res.repair.values()),
    }
    return SimOutcome(
        solved=res.accepted is not None,
        tests_used=len(res.tested),
        n_patches=len(eng.patches),
        oracle_any_passing=oracle,
        repair=res.repair,
        author_correct=bool(res.accepted and true in res.accepted.targets),
        order=order,
        extras=extras,
    )


def _m(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def study(cfg: SimConfig | None = None, n: int = 300, seed: int = 7) -> dict:
    cfg = cfg or SimConfig()
    out = {"config": cfg.__dict__ | {"skill_range": list(cfg.skill_range)}, "n_worlds": n, "synthetic": True}
    for order in ("price", "proposal", "random"):
        rows = [run_once(cfg, seed * 100000 + i, order) for i in range(n)]
        solvable = [r for r in rows if r.oracle_any_passing]
        out[order] = {
            "solve_rate": sum(r.solved for r in rows) / n,
            "solve_rate_when_solvable": sum(r.solved for r in solvable) / max(len(solvable), 1),
            "mean_tests_when_solved": statistics.mean([r.tests_used for r in rows if r.solved]) if any(r.solved for r in rows) else None,
            "mean_patches": statistics.mean(r.n_patches for r in rows),
            "info_credit_true_signal_agents": _m([r.extras["info_true_signal_mean"] for r in rows]),
            "info_credit_false_signal_agents": _m([r.extras["info_false_signal_mean"] for r in rows]),
            "repair_share_to_true_signal_agents": sum(r.extras["repair_to_true_signal"] for r in rows)
            / max(sum(r.extras["repair_total"] for r in rows), 1e-9),
            "n_solvable": len(solvable),
        }
    return out
