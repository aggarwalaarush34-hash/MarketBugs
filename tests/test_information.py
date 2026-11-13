import itertools
import math
import random

import pytest

from marketdebug.evidence import EvidenceBoard, Report
from marketdebug.information import (
    CoalitionValue, information_rewards, log_score, marginal_by_order, marginal_rewards,
    pivotal_rewards, quadratic_score, shapley_additive_approx, shapley_rewards,
)


def players(names):
    return {a: [Report(f"r{i}", a, i, f"f_{a}")] for i, a in enumerate(names)}


def table_value(table: dict[frozenset, float], rule="lift"):
    """Value function from a coalition -> forecast table (facts are f_<agent>)."""
    if rule == "lift":
        base = table[frozenset()]
        return lambda facts: table[frozenset(f[2:] for f in facts)] - base
    return CoalitionValue(lambda facts: table[frozenset(f[2:] for f in facts)], x=1, rule=rule)


def test_scoring_rules():
    assert quadratic_score(0.4, 1) == pytest.approx(0.64)
    assert quadratic_score(0.4, 0) == pytest.approx(0.84)
    assert log_score(0.5, 1) == pytest.approx(math.log(0.5))
    assert log_score(0.0, 1) > -100  # clipped, finite


def test_design_page_staircase_marginal_rewards():
    # forecasts 0.40 -> 0.55 (A) -> 0.70 (B) -> 0.95 (C), outcome x = 1
    seq = {(): 0.40, ("A",): 0.55, ("A", "B"): 0.70, ("A", "B", "C"): 0.95}
    fc = lambda facts: seq[tuple(sorted(f[2:] for f in facts))]
    reps = [Report("r0", "A", 0, "f_A"), Report("r1", "B", 1, "f_B"), Report("r2", "C", 2, "f_C")]
    quad = marginal_rewards(reps, CoalitionValue(fc, 1, "quadratic"))
    assert quad == pytest.approx({"A": 0.1575, "B": 0.1125, "C": 0.0875})
    log = marginal_rewards(reps, CoalitionValue(fc, 1, "log"))
    assert log["A"] == pytest.approx(math.log(0.55 / 0.40))
    assert log["C"] == pytest.approx(math.log(0.95 / 0.70))
    wrong = marginal_rewards(reps, CoalitionValue(fc, 0, "quadratic"))
    assert all(v < 0 for v in wrong.values())


def random_table(names, rng):
    t = {}
    for r in range(len(names) + 1):
        for c in itertools.combinations(names, r):
            t[frozenset(c)] = rng.uniform(0.05, 0.95)
    return t


def test_shapley_axioms_on_random_games():
    rng = random.Random(0)
    names = ["A", "B", "C", "D"]
    for _ in range(20):
        t = random_table(names, rng)
        v = CoalitionValue(lambda facts: t[frozenset(f[2:] for f in facts)], 1, "log")
        pl = players(names)
        phi = shapley_rewards(pl, v)
        # efficiency
        assert sum(phi.values()) == pytest.approx(v([f"f_{a}" for a in names]))
        # exact == average of marginal over all orders
        orders = marginal_by_order(pl, v)
        for a in names:
            assert phi[a] == pytest.approx(sum(orders[a]) / len(orders[a]))
        assert len(orders["A"]) == math.factorial(4)


def test_shapley_symmetry_and_null_player():
    t = {}
    for r in range(4):
        for c in itertools.combinations("ABC", r):
            s = set(c)
            t[frozenset(c)] = 0.2 + 0.3 * (("A" in s) and ("B" in s))  # A,B complementary; C null
    v = table_value(t)
    phi = shapley_rewards(players("ABC"), v)
    assert phi["A"] == pytest.approx(phi["B"]) == pytest.approx(0.15)
    assert phi["C"] == pytest.approx(0.0)
    piv = pivotal_rewards(players("ABC"), v)
    assert piv["A"] == pytest.approx(0.3) and piv["C"] == pytest.approx(0.0)
    assert sum(piv.values()) > sum(phi.values())  # pivotal over-pays complements


def test_marginal_is_order_dependent_for_complements():
    t = {frozenset(): 0.0, frozenset("A"): 0.0, frozenset("B"): 0.0, frozenset("AB"): 1.0}
    v = table_value(t)
    o = marginal_by_order(players("AB"), v)
    assert sorted(o["A"]) == [0.0, 1.0]


def test_sampled_shapley_close_to_exact():
    rng = random.Random(5)
    names = list("ABCDEFGHI")  # 9 > exact_up_to=8, forces sampling
    weights = {a: rng.random() for a in names}
    v = lambda facts: sum(weights[f[2:]] for f in facts) ** 1.2
    phi = shapley_rewards(players(names), v, samples=4000, rng=random.Random(1))
    exact = shapley_rewards(players(names), v, exact_up_to=9)
    for a in names:
        assert phi[a] == pytest.approx(exact[a], abs=0.03)


def test_additive_approx_exact_for_additive_games():
    w = {"A": 0.1, "B": -0.05, "C": 0.3}
    v = lambda facts: sum(w[f[2:]] for f in facts)
    pl = players("ABC")
    assert shapley_additive_approx(pl, v) == pytest.approx(shapley_rewards(pl, v))


def test_information_rewards_dispatch_and_scale():
    t = random_table(list("AB"), random.Random(2))
    v = table_value(t)
    pl = players("AB")
    reps = [r for rs in pl.values() for r in rs]
    out = information_rewards("shapley", reps, pl, v, scale=10)
    assert sum(out.values()) == pytest.approx(10 * v(["f_A", "f_B"]))
    assert information_rewards("none", reps, pl, v) == {}
    with pytest.raises(ValueError):
        information_rewards("vibes", reps, pl, v)


def test_coalition_value_caches():
    calls = []
    v = CoalitionValue(lambda facts: (calls.append(facts), 0.5)[1], 1, "log")
    v(["x"]); v(["x"]); v(["x"])
    assert v.calls == 2  # empty set + {x}


def test_board_denies_copies():
    board = EvidenceBoard()
    r1 = board.post("a", "parser_odd_value")
    r2 = board.post("b", "parser_odd_value")
    r3 = board.post("b", "caller_expects_other")
    assert r1.counts and not r2.counts and r3.counts
    assert [r.agent for r in board.counted()] == ["a", "b"]
    assert set(board.by_agent()) == {"a", "b"}
    canon = EvidenceBoard(canonicalize=lambda fact, text: fact.lower().strip())
    canon.post("a", "Parser Odd ")
    assert not canon.post("b", "parser odd").counts
