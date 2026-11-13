"""Every experiment in the study. Each function returns a JSON-able dict.

Primary (confirmatory) comparison, fixed in PROTOCOL.md before the frozen run:
  top-1 of `market` (frozen config, capital carried dev -> test) minus top-1 of
  `all_notes` on the held-out split, two-stage repository-cluster bootstrap.
Everything else is exploratory: effect sizes and intervals, no significance claims.
"""

from __future__ import annotations

import itertools
import math
import statistics
from dataclasses import replace

from ..agents.cached import mask_of_facts, note_fact
from ..bench.baselines import Baselines
from ..bench.localization import MarketConfig, gold_value, new_exchange, run_split
from ..data import AGENTS, FULL, Bug, bit
from ..evidence import Report
from ..information import (
    CoalitionValue, marginal_by_order, pivotal_rewards, shapley_additive_approx, shapley_rewards,
)
from .bootstrap import paired
from .metrics import reliability, score, summarize

PRIMARY = ("market", "all_notes")


# ---------------------------------------------------------------- helpers
def market_forecasts(bugs: list[Bug], cfg: MarketConfig, carry_from: list[Bug] | None = None):
    """Run the market over `bugs`; if `carry_from` is given, run it first on the same exchange."""
    ex = new_exchange(cfg)
    if carry_from:
        run_split(carry_from, cfg, ex)
    res, ex = run_split(bugs, cfg, ex)
    return {r.bug_id: r for r in res}, ex


def fresh_market_forecasts(bugs: list[Bug], cfg: MarketConfig):
    out = {}
    for b in bugs:
        res, _ = run_split([b], cfg)
        out[b.id] = res[0]
    return out


def score_methods(bugs: list[Bug], dists: dict[str, dict[str, list[float]]]) -> dict:
    per = {m: [score(dists[m][b.id], b.gold) for b in bugs] for m in dists}
    return {m: summarize(rows) for m, rows in per.items()}, per


def per_repo(bugs: list[Bug], per: dict[str, list[dict]], methods) -> dict:
    out = {}
    for repo in sorted({b.repo for b in bugs}):
        idx = [i for i, b in enumerate(bugs) if b.repo == repo]
        out[repo] = {m: summarize([per[m][i] for i in idx]) for m in methods}
    return out


def per_stratum(bugs: list[Bug], per: dict[str, list[dict]], methods) -> dict:
    out = {}
    for s in sorted({b.stratum or "?" for b in bugs}):
        idx = [i for i, b in enumerate(bugs) if (b.stratum or "?") == s]
        out[s] = {m: summarize([per[m][i] for i in idx]) for m in methods}
    return out


# ---------------------------------------------------------------- E1 main
def main_comparison(dev: list[Bug], test: list[Bug], cfg: MarketConfig, reps: int = 10000) -> dict:
    base = Baselines(dev)
    dists: dict[str, dict[str, list[float]]] = {}
    for b in test:
        for m, p in base.forecasts(b).items():
            dists.setdefault(m, {})[b.id] = p
    carried, ex = market_forecasts(test, cfg, carry_from=dev)
    dists["market"] = {k: r.prices for k, r in carried.items()}
    cold, _ = market_forecasts(test, cfg)
    dists["market.cold_start"] = {k: r.prices for k, r in cold.items()}
    fresh = fresh_market_forecasts(test, cfg)
    dists["market.fresh_per_bug"] = {k: r.prices for k, r in fresh.items()}

    summary, per = score_methods(test, dists)
    clusters = [b.repo for b in test]
    comps = {}
    for a, b_ in [PRIMARY, ("market", "all_notes.T"), ("market", "hedge.T"), ("market", "single.static"),
                  ("market.fresh_per_bug", "all_notes"), ("market", "no_notes")]:
        comps[f"{a} - {b_}"] = {
            "top1": paired([r["top1"] for r in per[a]], [r["top1"] for r in per[b_]], clusters, reps),
            "nll": paired([r["nll"] for r in per[a]], [r["nll"] for r in per[b_]], clusters, reps),
        }
    same_pick = sum(
        1 for b in test
        if max(range(b.k), key=lambda i: (dists["market"][b.id][i], -i))
        == max(range(b.k), key=lambda i: (dists["all_notes"][b.id][i], -i))
    )
    rel = {m: reliability([r["conf"] for r in per[m]], [r["correct"] for r in per[m]])
           for m in ("market", "all_notes", "all_notes.T", "hedge.T")}
    accts = ex.accounts
    return {
        "config": cfg.to_dict(),
        "baseline_params": base.params(),
        "n": len(test),
        "summary": summary,
        "comparisons": comps,
        "primary": comps[f"{PRIMARY[0]} - {PRIMARY[1]}"]["top1"],
        "market_same_top1_as_all_notes": same_pick,
        "per_repo": per_repo(test, per, ["market", "all_notes", "all_notes.T", "hedge.T"]),
        "per_stratum": per_stratum(test, per, ["market", "all_notes", "all_notes.T"]),
        "reliability": rel,
        "ledgers_after_test": {
            a: {"capital": c.capital, "trading": c.trading, "information": c.information,
                "repair": c.repair, "floor_topups": c.floor_topups}
            for a, c in accts.items()
        },
        "maker_loss_total": sum(r.maker_loss for r in carried.values()),
        "voided_markets": sum(r.voided for r in carried.values()),
        "denied_reveals": sum(len(r.denied_reveals) for r in carried.values()),
        "audit": ex.audit(),
    }


# ---------------------------------------------------------------- E2 ledger ablation
LEDGER_VARIANTS = {
    "trading_only": dict(capital_from=("trading",), info_mechanism="auto"),
    "trading+shapley": dict(capital_from=("trading", "information"), info_mechanism="auto"),
    "trading+marginal": dict(capital_from=("trading", "information"), info_mechanism="marginal"),
    "trading+pivotal": dict(capital_from=("trading", "information"), info_mechanism="pivotal"),
}


def ledger_ablation(dev: list[Bug], test: list[Bug], cfg: MarketConfig, reps: int = 2000) -> dict:
    out = {"dev_only": {}}
    # on dev every coalition is cached, so all four variants run
    for name, kw in LEDGER_VARIANTS.items():
        res, ex = market_forecasts(dev, replace(cfg, **kw))
        out["dev_only"][name] = {
            "summary": summarize([score(res[b.id].prices, b.gold) for b in dev]),
            "final_capital": {a: acct.capital for a, acct in ex.accounts.items()},
        }
    base_rows = None
    for name, kw in LEDGER_VARIANTS.items():
        c = replace(cfg, **kw)
        try:
            res, ex = market_forecasts(test, c, carry_from=dev)
        except Exception as e:  # pivotal on test needs coalitions that were never cached
            out[name] = {"skipped": f"{type(e).__name__}: {e}"}
            continue
        rows = [score(res[b.id].prices, b.gold) for b in test]
        if base_rows is None:
            base_rows = rows
        out[name] = {
            "summary": summarize(rows),
            "vs_trading_only_top1": paired([r["top1"] for r in rows], [r["top1"] for r in base_rows],
                                           [b.repo for b in test], reps),
            "vs_trading_only_nll": paired([r["nll"] for r in rows], [r["nll"] for r in base_rows],
                                          [b.repo for b in test], reps),
            "final_capital": {a: acct.capital for a, acct in ex.accounts.items()},
            "information_ledger": {a: acct.information for a, acct in ex.accounts.items()},
        }
    return out


# ---------------------------------------------------------------- E3 credit properties (dev, exact)
def players_for(bug: Bug, agents=AGENTS) -> dict[str, list[Report]]:
    return {a: [Report(f"r{i}", a, 0, note_fact(a))] for i, a in enumerate(agents)}


def credit_properties(dev: list[Bug], rule: str = "log") -> dict:
    rows = []
    for b in dev:
        if not (b.gold_in_pool and b.has_full_table):
            continue
        v = gold_value(b, rule)
        pl = players_for(b)
        exact = shapley_rewards(pl, v)
        approx = shapley_additive_approx(pl, v)
        piv = pivotal_rewards(pl, v)
        orders = marginal_by_order(pl, v)
        total = v([note_fact(a) for a in AGENTS])
        rows.append({
            "id": b.id,
            "v_full": total,
            "shapley": exact,
            "shapley_approx": approx,
            "pivotal": piv,
            "pivotal_sum_minus_total": sum(piv.values()) - total,
            "shapley_sum_minus_total": sum(exact.values()) - total,
            "marginal_range": {a: max(x) - min(x) for a, x in orders.items()},
            "marginal_sd": {a: statistics.pstdev(x) for a, x in orders.items()},
            "rank_flip_orders": _rank_flip_fraction(orders),
        })
    n = len(rows)

    def avg(f):
        return sum(f(r) for r in rows) / n if n else None

    mean_shapley = {a: avg(lambda r, a=a: r["shapley"][a]) for a in AGENTS}
    approx_err = [abs(r["shapley_approx"][a] - r["shapley"][a]) for r in rows for a in AGENTS]
    approx_rank_agree = sum(
        1 for r in rows
        if _argmax(r["shapley"]) == _argmax(r["shapley_approx"])
    )
    return {
        "n_bugs": n,
        "scoring_rule": rule,
        "mean_shapley": mean_shapley,
        "mean_pivotal": {a: avg(lambda r, a=a: r["pivotal"][a]) for a in AGENTS},
        "mean_marginal_range": {a: avg(lambda r, a=a: r["marginal_range"][a]) for a in AGENTS},
        "mean_abs_shapley": avg(lambda r: sum(abs(x) for x in r["shapley"].values()) / len(AGENTS)),
        "max_shapley_budget_error": max(abs(r["shapley_sum_minus_total"]) for r in rows) if rows else None,
        "mean_pivotal_budget_gap": avg(lambda r: r["pivotal_sum_minus_total"]),
        "share_pivotal_overpays": avg(lambda r: float(r["pivotal_sum_minus_total"] > 1e-9)),
        "mean_rank_flip_fraction": avg(lambda r: r["rank_flip_orders"]),
        "approx_mean_abs_error": sum(approx_err) / len(approx_err) if approx_err else None,
        "approx_top_agent_agreement": approx_rank_agree / n if n else None,
        "per_bug": rows,
    }


def _argmax(d: dict) -> str:
    return max(sorted(d), key=lambda k: d[k])


def _rank_flip_fraction(orders: dict[str, list[float]]) -> float:
    """Fraction of reveal orders whose top-credited agent differs from the most common one."""
    names = sorted(orders)
    n = len(orders[names[0]])
    tops = [max(names, key=lambda a: orders[a][i]) for i in range(n)]
    mode = max(set(tops), key=tops.count)
    return sum(t != mode for t in tops) / n


# ---------------------------------------------------------------- E4 copycat
def copycat(dev: list[Bug], cfg: MarketConfig, target: str = "graph", rule: str = "log") -> dict:
    """With the board's dedupe rule the copycat earns nothing; without it, Shapley splits credit."""
    with_rule, without_rule, original_alone = [], [], []
    for b in dev:
        if not (b.gold_in_pool and b.has_full_table):
            continue
        v = gold_value(b, rule)
        pl = players_for(b)
        original_alone.append(shapley_rewards(pl, v)[target])
        pl_dup = dict(pl)
        pl_dup["copycat"] = [Report("rc", "copycat", 1, note_fact(target))]
        sh = shapley_rewards(pl_dup, v)
        without_rule.append((sh[target], sh["copycat"]))
    res, _ = market_forecasts(dev, replace(cfg, copycat_of=target))
    cc = f"copycat_{target}"
    for r in res.values():
        with_rule.append((r.information.get(target, 0.0), r.information.get(cc, 0.0)))
    n = len(without_rule)
    return {
        "target": target,
        "n_bugs": n,
        "no_dedupe_mean_original": sum(x for x, _ in without_rule) / n,
        "no_dedupe_mean_copycat": sum(y for _, y in without_rule) / n,
        "original_without_copycat": sum(original_alone) / n,
        "with_dedupe_market_mean_original_credits": sum(x for x, _ in with_rule) / len(with_rule),
        "with_dedupe_market_mean_copycat_credits": sum(y for _, y in with_rule) / len(with_rule),
        "copycat_denied_every_time": all(y == 0.0 for _, y in with_rule),
    }


# ---------------------------------------------------------------- E5 conditioning on public notes (dev)
def conditioning(dev: list[Bug], cfg: MarketConfig) -> dict:
    out = {}
    for flag in (False, True):
        for pol in ("trade_then_reveal", "reveal_first"):
            c = replace(cfg, condition_on_public=flag, reveal_policy=pol)
            res, _ = market_forecasts(dev, c)
            out[f"condition={flag},policy={pol}"] = summarize([score(res[b.id].prices, b.gold) for b in dev])
    return out


# ---------------------------------------------------------------- E6 reveal policy / hoarding
def hoarding(dev: list[Bug], test: list[Bug], cfg: MarketConfig) -> dict:
    out = {}
    for pol in ("trade_then_reveal", "reveal_first", "hoard"):
        c = replace(cfg, reveal_policy=pol, capital_from=("trading", "information"))
        res, ex = market_forecasts(test, c, carry_from=dev)
        out[pol] = {
            "summary": summarize([score(res[b.id].prices, b.gold) for b in test]),
            "information_ledger": {a: x.information for a, x in ex.accounts.items()},
            "final_capital": {a: x.capital for a, x in ex.accounts.items()},
        }
    return out


# ---------------------------------------------------------------- E7 robustness
def robustness(dev: list[Bug], test: list[Bug], cfg: MarketConfig, seeds: int = 20) -> dict:
    """Sensitivity to turn order (seeds), bug order, and each market parameter (one at a time)."""
    def run(c, order_seed=None):
        ex = new_exchange(c)
        run_split(dev, c, ex)
        res, _ = run_split(test, c, ex, order_seed=order_seed)
        by = {r.bug_id: r for r in res}
        return summarize([score(by[b.id].prices, b.gold) for b in test])

    seed_rows = [run(replace(cfg, seed=s)) for s in range(seeds)]
    order_rows = [run(cfg, order_seed=s) for s in range(seeds)]

    def band(rows, key):
        xs = [r[key] for r in rows]
        return {"mean": statistics.mean(xs), "min": min(xs), "max": max(xs), "sd": statistics.pstdev(xs)}

    grid = {
        "b": [10, 25, 50, 100, 200],
        "per_turn_fraction": [0.01, 0.0333, 0.0667, 0.15, 0.3],
        "rounds": [1, 2, 3, 5, 8],
        "open_at": ["no_notes", "uniform"],
    }
    one_at_a_time = {
        k: {str(v): run(replace(cfg, **{k: v})) for v in vals} for k, vals in grid.items()
    }
    return {
        "turn_order_seeds": {"top1": band(seed_rows, "top1"), "nll": band(seed_rows, "nll")},
        "bug_order_seeds": {"top1": band(order_rows, "top1"), "nll": band(order_rows, "nll")},
        "one_at_a_time": one_at_a_time,
    }


# ---------------------------------------------------------------- freeze (dev only)
FREEZE_GRID = {
    "b": [10, 25, 50, 100, 200],
    "per_turn_fraction": [0.0333, 0.0667, 0.15],
    "rounds": [1, 3, 5],
    "open_at": ["no_notes", "uniform"],
}


def select_on_dev(dev: list[Bug], base: MarketConfig = MarketConfig()) -> dict:
    """Grid search on dev only; objective: dev gold NLL of the carried-capital market.

    Ties (within 1e-6) break toward the earlier grid entry, i.e. smaller b,
    fewer rounds and smaller budgets.
    """
    trials = []
    for b, f, r, o in itertools.product(*FREEZE_GRID.values()):
        c = replace(base, b=b, per_turn_fraction=f, rounds=r, open_at=o)
        res, _ = run_split(dev, c)
        by = {x.bug_id: x for x in res}
        s = summarize([score(by[x.id].prices, x.gold) for x in dev])
        trials.append({"config": c.to_dict(), "dev": s})
    best = min(range(len(trials)), key=lambda i: (round(trials[i]["dev"]["nll"], 6), i))
    return {"objective": "dev gold NLL", "grid": FREEZE_GRID, "trials": trials, "selected": trials[best]}


def all_experiments(dev, test, cfg, reps=10000):
    return {
        "main": main_comparison(dev, test, cfg, reps),
        "ledger_ablation": ledger_ablation(dev, test, cfg),
        "credit_properties": credit_properties(dev),
        "copycat": copycat(dev, cfg),
        "conditioning_dev": conditioning(dev, cfg),
        "hoarding": hoarding(dev, test, cfg),
        "robustness": robustness(dev, test, cfg),
    }
