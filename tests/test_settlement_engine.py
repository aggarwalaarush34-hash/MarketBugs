import pytest

from marketdebug.engine import Engine, EngineConfig
from marketdebug.exchange import Exchange
from marketdebug.protocol import OpenLocationMarket, ProposePatch, Reveal, Trade, TradeDistribution
from marketdebug.settlement import Patch, SettlementConfig, settle
from marketdebug.evidence import EvidenceBoard


class Scripted:
    def __init__(self, name, script):
        self.name = name
        self.script = script  # step -> list of actions

    def act(self, obs):
        return list(self.script.get(obs.step, []))


def setup_patches(prices):
    ex = Exchange()
    ex.register("author0"); ex.register("author1"); ex.register("author2"); ex.register("t")
    patches = []
    for i, p in enumerate(prices):
        pt = Patch(f"P{i}", f"author{i}", 0, frozenset({f"f{i}"}))
        ex.open_market(pt.market_id, "pass?", kind="patch", target=pt.patch_id)
        ex.trade_to_belief("t", pt.market_id, p)
        patches.append(pt)
    return ex, patches


def test_tests_run_in_price_order_and_stop_at_first_pass():
    ex, patches = setup_patches([0.2, 0.9, 0.6])
    ran = []
    res = settle(ex, EvidenceBoard(), patches, lambda p: (ran.append(p.patch_id), p.patch_id in ("P1", "P2"))[1],
                 lambda mid, f: 0.5, SettlementConfig(test_budget=5))
    assert ran == ["P1"]
    assert res.accepted.patch_id == "P1"
    assert res.resolutions["patch:P1"] == "YES"
    assert res.resolutions["patch:P0"] == "VOID" and res.resolutions["patch:P2"] == "VOID"
    assert ex.audit()["balanced"]


def test_given_order_and_budget():
    ex, patches = setup_patches([0.2, 0.9, 0.6])
    ran = []
    res = settle(ex, EvidenceBoard(), patches, lambda p: (ran.append(p.patch_id), False)[1],
                 lambda mid, f: 0.5, SettlementConfig(test_budget=2), order="given")
    assert ran == ["P0", "P1"]
    assert res.accepted is None
    with pytest.raises(ValueError):
        settle(ex, EvidenceBoard(), [], lambda p: True, lambda m, f: 0.5, order="sideways")


def test_full_episode_root_cause_info_and_repair():
    a = Scripted("a", {0: [OpenLocationMarket("parse_input"), Trade("loc:parse_input", 0.85), Reveal("f1")],
                       1: [ProposePatch(("parse_input",), ("f1", "f2"))]})
    b = Scripted("b", {0: [Reveal("f1"), Trade("loc:parse_input", 0.3)]})  # copycat of f1
    c = Scripted("c", {0: [Reveal("f2")], 1: [Trade("loc:parse_input", 0.9)]})
    fc = lambda mid, facts: 0.4 + 0.25 * ("f1" in facts) + 0.3 * ("f2" in facts)
    eng = Engine([a, b, c], lambda p: "parse_input" in p.targets, fc,
                 EngineConfig(rounds=2, shuffle_turn_order=False,
                              settlement=SettlementConfig(repair_bounty=200, cite_share=0.3)))
    res = eng.run()
    assert res.accepted.author == "a"
    assert res.resolutions["loc:parse_input"] == "YES"
    assert "b" not in res.information  # copy denied
    assert res.information["a"] > 0 and res.information["c"] > 0
    assert res.repair["a"] == pytest.approx(140 + 30)  # author share + half the cite pool (a found f1)
    assert res.repair["c"] == pytest.approx(30)
    assert sum(res.repair.values()) == pytest.approx(200)
    assert eng.ex.audit()["balanced"]


def test_no_passing_patch_voids_root_markets():
    a = Scripted("a", {0: [OpenLocationMarket("x"), Trade("loc:x", 0.9), ProposePatch(("x",))]})
    eng = Engine([a], lambda p: False, lambda m, f: 0.5, EngineConfig(rounds=1))
    res = eng.run()
    assert res.accepted is None
    assert res.resolutions["loc:x"] == "VOID"
    assert res.repair == {}
    assert eng.ex.accounts["a"].capital == pytest.approx(1000)  # refunded


def test_engine_logs_bad_actions_and_caps_actions():
    a = Scripted("a", {0: [Trade("nope", 0.5)] + [Reveal(f"f{i}") for i in range(50)]})
    eng = Engine([a], cfg=EngineConfig(rounds=1, max_actions_per_turn=5))
    eng.play()
    assert eng.log.of_type("error")
    assert len(eng.board.reports) == 4
    with pytest.raises(ValueError):
        eng.run()  # default settlement needs run_tests + forecaster
    with pytest.raises(ValueError):
        Engine([a, Scripted("a", {})])


def test_trade_distribution_action():
    from marketdebug.lmsr import LMSRMarket

    ex = Exchange()
    ex.add_market(LMSRMarket.opening_at("bug", "q", [0.25] * 4, 20))
    ex.markets["bug"].meta["kind"] = "localization"
    a = Scripted("a", {0: [TradeDistribution("bug", (0.7, 0.1, 0.1, 0.1), budget=10.0)]})
    Engine([a], cfg=EngineConfig(rounds=1), exchange=ex).play()
    assert ex.markets["bug"].price(0) > 0.25
    assert ex.accounts["a"].capital >= 990 - 1e-9
