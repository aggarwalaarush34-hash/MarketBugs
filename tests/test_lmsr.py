import math
import random

import pytest

from marketdebug.lmsr import LMSRMarket, MarketClosed, geometric_mix, normalize


def binary(b=100.0):
    return LMSRMarket("m", "q?", ["NO", "YES"], b)


def test_uniform_start_and_prices_sum_to_one():
    m = LMSRMarket("m", "q", [str(i) for i in range(5)], 50)
    assert all(abs(p - 0.2) < 1e-12 for p in m.prices())
    m.buy("a", 2, 30)
    assert abs(sum(m.prices()) - 1) < 1e-12
    assert m.price(2) > 0.2


def test_worked_example_from_design_page():
    # market 0.35 -> belief 0.85 at b = 100: ~235.4 shares for ~146.6 credits
    m = binary(100)
    m.buy("maker", 1, m.shares_to_reach(1, 0.35))
    m.positions.clear(); m.spent.clear()
    shares = m.shares_to_reach(1, 0.85)
    cost = m.quote(1, shares)
    assert shares == pytest.approx(235.4, abs=0.1)
    assert cost == pytest.approx(100 * math.log(0.65 / 0.15), rel=1e-9)
    assert cost == pytest.approx(146.6, abs=0.1)
    m.buy("agent7", 1, shares)
    assert m.price(1) == pytest.approx(0.85)


def test_shares_to_prices_hits_target_exactly_and_is_buy_only():
    rng = random.Random(1)
    m = LMSRMarket("m", "q", [str(i) for i in range(6)], 20)
    for _ in range(20):
        target = normalize([rng.random() + 1e-3 for _ in range(6)])
        d = m.shares_to_prices(target)
        assert min(d) == pytest.approx(0.0, abs=1e-9)
        m.buy_vector("a", d)
        for p, t in zip(m.prices(), target):
            assert p == pytest.approx(t, rel=1e-9)


def test_round_trip_costs_nothing():
    m = binary(30)
    c1 = m.quote(1, 40)
    m.buy("a", 1, 40)
    c2 = m.quote(1, -40)
    assert c1 + c2 == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("uniform", [True, False])
def test_maker_loss_never_exceeds_bound(uniform):
    rng = random.Random(7)
    for trial in range(50):
        k = rng.randint(2, 8)
        b = rng.choice([5, 20, 100])
        if uniform:
            m = LMSRMarket("m", "q", [str(i) for i in range(k)], b)
        else:
            m = LMSRMarket.opening_at("m", "q", [rng.random() + 0.01 for _ in range(k)], b)
        for _ in range(rng.randint(1, 15)):
            m.buy_vector(f"a{rng.randint(0, 3)}", [rng.random() * 50 for _ in range(k)])
        win = rng.randrange(k)
        m.resolve(win)
        assert m.maker_loss() <= m.max_maker_loss + 1e-9
    if uniform:
        assert LMSRMarket("x", "q", ["a", "b", "c"], 10).max_maker_loss == pytest.approx(10 * math.log(3))


def test_pnl_sums_to_minus_maker_profit():
    m = LMSRMarket.opening_at("m", "q", [0.5, 0.3, 0.2], 25)
    m.buy("a", 0, 10); m.buy("b", 2, 30); m.buy("a", 1, 5)
    m.resolve(2)
    assert sum(m.pnl().values()) == pytest.approx(m.maker_loss(), abs=1e-9)


def test_weighted_resolution_and_void():
    m = LMSRMarket("m", "q", ["x", "y", "z"], 10)
    m.buy("a", 0, 10)
    m.buy("b", 1, 10)
    pay = m.resolve_weighted([0.5, 0.5, 0.0])
    assert pay["a"] == pytest.approx(5) and pay["b"] == pytest.approx(5)
    with pytest.raises(MarketClosed):
        m.buy("a", 0, 1)
    v = LMSRMarket("v", "q", ["x", "y"], 10)
    v.buy("a", 0, 3)
    refund = v.void()
    assert refund["a"] == pytest.approx(v.spent["a"])
    assert v.pnl() == {"a": 0.0}
    assert v.maker_loss() == 0.0


def test_bad_inputs():
    with pytest.raises(ValueError):
        LMSRMarket("m", "q", ["only"], 10)
    with pytest.raises(ValueError):
        LMSRMarket("m", "q", ["a", "b"], 0)
    m = binary()
    with pytest.raises(ValueError):
        m.resolve_weighted([0.7, 0.7])
    with pytest.raises(ValueError):
        m.shares_to_reach(1, 1.0)
    with pytest.raises(ValueError):
        m.pnl()


def test_opening_at_and_geometric_mix():
    p0 = [0.6, 0.3, 0.1]
    m = LMSRMarket.opening_at("m", "q", p0, 40)
    assert m.prices() == pytest.approx(p0)
    assert m.opening_prices() == pytest.approx(p0)
    assert geometric_mix(p0, [0.1, 0.1, 0.8], 0.0) == pytest.approx(p0)
    assert geometric_mix(p0, [0.1, 0.1, 0.8], 1.0) == pytest.approx([0.1, 0.1, 0.8])
