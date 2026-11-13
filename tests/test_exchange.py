import random

import pytest

from marketdebug.exchange import Exchange, InsufficientFunds
from marketdebug.lmsr import LMSRMarket


def ex_with(**kw):
    ex = Exchange(**kw)
    for a in ("a", "b", "c"):
        ex.register(a)
    return ex


def test_insufficient_funds():
    ex = ex_with(starting_capital=10)
    ex.open_market("m", "q", b=100)
    with pytest.raises(InsufficientFunds):
        ex.buy("a", "m", 1, 1000)


def test_trade_toward_respects_budget_and_moves_toward_belief():
    ex = ex_with(default_b=50)
    ex.add_market(LMSRMarket.opening_at("m", "q", [0.25] * 4, 50))
    belief = [0.7, 0.1, 0.1, 0.1]
    t = ex.trade_toward("a", "m", belief, budget=5.0)
    assert t.cost <= 5.0 + 1e-9
    p = ex.markets["m"].prices()
    assert 0.25 < p[0] < 0.7
    assert ex.accounts["a"].capital == pytest.approx(1000 - t.cost)


def test_unlimited_budget_sets_prices_to_belief():
    ex = ex_with(default_b=10, max_fraction_per_trade=1.0)
    ex.add_market(LMSRMarket.opening_at("m", "q", [0.25] * 4, 10))
    belief = [0.4, 0.3, 0.2, 0.1]
    ex.trade_toward("a", "m", belief)
    assert ex.markets["m"].prices() == pytest.approx(belief, rel=1e-6)


def test_binary_trade_to_belief_buys_no_when_price_too_high():
    ex = ex_with()
    ex.open_market("m", "q", b=100)
    ex.trade_to_belief("a", "m", 0.8)
    assert ex.markets["m"].price(1) > 0.5
    t = ex.trade_to_belief("b", "m", 0.3)
    assert t.outcome == 0
    assert ex.markets["m"].price(1) < 0.8


def test_trade_to_belief_rejects_multi_outcome():
    ex = ex_with()
    ex.open_market("m", "q", ["x", "y", "z"])
    with pytest.raises(ValueError):
        ex.trade_to_belief("a", "m", 0.5)


def test_books_balance_after_random_activity():
    rng = random.Random(3)
    ex = ex_with(capital_floor=1.0, capital_from=("trading", "information", "repair"))
    for i in range(30):
        k = rng.randint(2, 5)
        mid = f"m{i}"
        ex.add_market(LMSRMarket.opening_at(mid, "q", [rng.random() + 0.05 for _ in range(k)], rng.choice([10, 50])))
        for _ in range(5):
            a = rng.choice("abc")
            ex.trade_toward(a, mid, [rng.random() + 0.01 for _ in range(k)])
        r = rng.random()
        if r < 0.15:
            ex.void(mid)
        else:
            ex.resolve(mid, rng.randrange(k))
        ex.book("information", {rng.choice("abc"): rng.uniform(-20, 20)})
        ex.book("repair", {rng.choice("abc"): rng.uniform(0, 30)})
    audit = ex.audit()
    assert audit["balanced"], audit
    assert audit["open_markets"] == []
    # traders' combined P&L is exactly what the maker paid out (zero-sum against the maker)
    total_trading = sum(a.trading for a in ex.accounts.values())
    assert total_trading == pytest.approx(sum(m.maker_loss() for m in ex.markets.values()), abs=1e-6)


def test_trading_pnl_equals_minus_maker_profit():
    ex = ex_with()
    ex.add_market(LMSRMarket.opening_at("m", "q", [0.5, 0.25, 0.25], 20))
    ex.trade_toward("a", "m", [0.8, 0.1, 0.1])
    ex.trade_toward("b", "m", [0.1, 0.1, 0.8])
    ex.resolve("m", 2)
    assert sum(a.trading for a in ex.accounts.values()) == pytest.approx(ex.markets["m"].maker_loss())


def test_ledger_routing_and_floor():
    ex = ex_with(capital_floor=5.0, capital_from=("trading", "information"))
    ex.book("information", {"a": 50.0})
    ex.book("repair", {"a": 70.0})
    acct = ex.accounts["a"]
    assert acct.information == 50 and acct.repair == 70
    assert acct.capital == pytest.approx(1050)  # repair not routed into capital
    ex.book("information", {"b": -2000.0})
    assert ex.accounts["b"].capital == 5.0
    assert ex.accounts["b"].floor_topups == pytest.approx(1005.0)  # from -1000 up to the floor of 5
    assert ex.audit()["balanced"]


def test_bad_configs():
    with pytest.raises(ValueError):
        Exchange(capital_from=("information",))
    with pytest.raises(ValueError):
        Exchange(capital_from=("trading", "fame"))
    ex = ex_with()
    with pytest.raises(ValueError):
        ex.book("trading", {"a": 1})
    ex.open_market("m", "q")
    with pytest.raises(ValueError):
        ex.add_market(LMSRMarket("m", "q"))


def test_market_cap():
    ex = ex_with(max_markets=2)
    ex.open_market("m1", "q"); ex.open_market("m2", "q")
    with pytest.raises(RuntimeError):
        ex.open_market("m3", "q")
