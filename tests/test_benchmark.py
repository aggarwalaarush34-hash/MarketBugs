"""Tests over the real cached Gemma data: validation, episodes, baselines, freeze contract."""

import json
import math
import random
from dataclasses import replace

import pytest

from marketdebug import freeze
from marketdebug.agents.cached import CachedGemmaAgent, mask_of_facts, note_fact
from marketdebug.analysis.bootstrap import cluster_bootstrap, instance_bootstrap, paired
from marketdebug.analysis.metrics import ece, fit_temperature, score, summarize, temper
from marketdebug.bench.baselines import Baselines, hedge_weights, linpool, logpool
from marketdebug.bench.localization import MarketConfig, run_episode, run_split, new_exchange
from marketdebug.data import AGENTS, FULL, Bug, bit, load_split, parse, validate


@pytest.fixture(scope="module")
def dev():
    return load_split("dev")


@pytest.fixture(scope="module")
def test_split():
    return load_split("test")


# ---------------------------------------------------------------- data
def test_real_data_validates(dev, test_split):
    assert len(dev) == 30 and len(test_split) == 210
    assert all(b.has_full_table for b in dev)
    assert sum(b.gold_in_pool for b in test_split) == 172
    assert {b.repo for b in test_split} == {"attrs", "boltons", "dateutil", "isodate", "more-itertools", "tomli", "toolz"}
    assert not {b.repo for b in dev} & {b.repo for b in test_split}  # repository-disjoint splits


def test_validation_catches_problems():
    good = {"id": "x", "repo": "r", "gold": [0], "forecasts": {str(m): [0.5, 0.5] for m in (0, 1, 2, 4, 8, 16, 31)}}
    assert validate([parse(good)]) == []
    bad = json.loads(json.dumps(good))
    bad["forecasts"]["1"] = [0.9, 0.9]
    bad["gold"] = [5]
    del bad["forecasts"]["31"]
    probs = validate([parse(bad), parse(good), parse(good)])
    text = " ".join(probs)
    assert "sum" in text and "out of range" in text and "missing" in text and "duplicate" in text


def test_mask_helpers():
    assert mask_of_facts([note_fact("trace"), note_fact("semantic"), "other"]) == bit("trace") | bit("semantic")
    assert mask_of_facts([note_fact(a) for a in AGENTS]) == FULL


# ---------------------------------------------------------------- episodes
def test_episode_books_balance_and_deterministic(dev):
    cfg = MarketConfig()
    r1, ex1 = run_split(dev, cfg)
    r2, _ = run_split(dev, cfg)
    assert [x.prices for x in r1] == [x.prices for x in r2]
    assert ex1.audit()["balanced"]
    assert all(not x.errors for x in r1)


def test_retrieval_miss_voids_market(test_split):
    miss = next(b for b in test_split if not b.gold_in_pool)
    ex = new_exchange(MarketConfig())
    res = run_episode(miss, ex, MarketConfig(), random.Random(0))
    assert res.voided and res.information == {}
    assert all(abs(a.capital - 1000) < 1e-6 for a in ex.accounts.values())


def test_auto_mechanism_selection(dev, test_split):
    rd = run_episode(next(b for b in dev if b.gold_in_pool), new_exchange(MarketConfig()), MarketConfig(), random.Random(0))
    rt = run_episode(next(b for b in test_split if b.gold_in_pool), new_exchange(MarketConfig()), MarketConfig(), random.Random(0))
    assert rd.info_mechanism == "shapley" and rt.info_mechanism == "shapley_approx"


def test_shapley_info_sums_to_total_value(dev):
    cfg = MarketConfig(info_scale=1.0)
    b = next(x for x in dev if x.gold_in_pool)
    res = run_episode(b, new_exchange(cfg), cfg, random.Random(0))
    total = math.log(b.gold_mass(FULL)) - math.log(b.gold_mass(0))
    assert sum(res.information.values()) == pytest.approx(total)


def test_copycat_gets_no_information_credit(dev):
    cfg = MarketConfig(copycat_of="graph")
    res, _ = run_split(dev, cfg)
    assert all(r.information.get("copycat_graph", 0.0) == 0.0 for r in res)
    assert any(("copycat_graph", note_fact("graph")) in r.denied_reveals for r in res)


def test_hoarding_earns_no_information(dev):
    res, ex = run_split(dev, MarketConfig(reveal_policy="hoard"))
    assert all(r.information == {} for r in res)


def test_capital_carries_across_bugs(dev):
    _, ex = run_split(dev, MarketConfig())
    caps = [a.capital for a in ex.accounts.values()]
    assert max(caps) - min(caps) > 1.0


def test_information_into_capital_changes_capital(dev):
    _, ex0 = run_split(dev, MarketConfig())
    _, ex1 = run_split(dev, MarketConfig(capital_from=("trading", "information")))
    assert any(abs(ex0.accounts[a].capital - ex1.accounts[a].capital) > 1e-6 for a in AGENTS)


def test_conditioning_uses_public_coalitions(dev):
    b = dev[0]
    ag = CachedGemmaAgent("trace", "trace", b, condition_on_public=True)
    ex = new_exchange(MarketConfig())
    from marketdebug.engine import Engine, EngineConfig
    from marketdebug.lmsr import LMSRMarket
    m = LMSRMarket.opening_at("bug:x", "q", list(b.forecast(0)), 50)
    m.meta["kind"] = "localization"
    ex.add_market(m)
    eng = Engine([ag, CachedGemmaAgent("graph", "graph", b, policy="reveal_first")],
                 cfg=EngineConfig(rounds=2, shuffle_turn_order=False), exchange=ex)
    eng.play()
    assert any(mask != bit("trace") for mask in ag.beliefs_used)


def test_config_roundtrip():
    c = MarketConfig(b=25, capital_from=("trading", "information"))
    assert MarketConfig.from_dict(json.loads(json.dumps(c.to_dict()))) == c


# ---------------------------------------------------------------- baselines + metrics
def test_pools_normalise():
    d = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]
    assert sum(logpool(d)) == pytest.approx(1) and sum(linpool(d)) == pytest.approx(1)
    assert linpool(d) == pytest.approx([0.4, 0.5, 0.1])


def test_baselines_fit_on_dev(dev):
    base = Baselines(dev)
    p = base.params()
    assert sum(p["hedge_weights"].values()) == pytest.approx(1)
    assert p["hedge_eta"] in (0.1, 0.3, 1.0, 3.0)
    f = base.forecasts(dev[0])
    assert {"all_notes", "all_notes.T", "hedge.T", "single.graph"} <= set(f)
    assert all(sum(v) == pytest.approx(1) for v in f.values())


def test_temperature_recovers_known_value():
    rng = random.Random(0)
    dists, golds = [], []
    for _ in range(400):
        logits = [rng.gauss(0, 2) for _ in range(5)]
        true = temper([math.exp(x) for x in logits], 1.0)
        g = rng.choices(range(5), weights=true)[0]
        dists.append(temper(true, 1 / 2.5))  # overconfident by factor 2.5
        golds.append([g])
    assert fit_temperature(dists, golds) == pytest.approx(2.5, rel=0.25)


def test_score_and_ece():
    s = score([0.1, 0.6, 0.3], [2])
    assert s["top1"] == 0 and s["top3"] == 1 and s["rr"] == pytest.approx(0.5)
    assert s["nll"] == pytest.approx(-math.log(0.3))
    miss = score([0.5, 0.5], [])
    assert miss["top1"] == 0 and miss["nll"] is None
    tie = score([0.5, 0.5], [0])
    assert tie["top1"] == 1  # ties break to lowest index
    assert ece([0.9, 0.9], [1, 0]) == pytest.approx(0.4)
    assert summarize([s, miss])["n_gold_in_pool"] == 1


def test_bootstrap_deterministic_and_sensible():
    diffs = [1, 0, 0, 1, 0, -1, 1, 0] * 10
    cl = [f"r{i % 4}" for i in range(len(diffs))]
    a = cluster_bootstrap(diffs, cl, reps=500)
    assert a == cluster_bootstrap(diffs, cl, reps=500)
    assert a["lo"] <= a["mean"] <= a["hi"]
    i = instance_bootstrap(diffs, reps=500)
    assert i["lo"] <= i["mean"] <= i["hi"]
    p = paired([1, None, 0], [0, 1, 0], ["a", "a", "b"], reps=100)
    assert p["n"] == 2 and p["wins"] == 1 and p["ties"] == 1


# ---------------------------------------------------------------- freeze contract
def test_freeze_detects_code_and_data_changes(tmp_path, monkeypatch, dev):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("x = 1\n")
    data = tmp_path / "dev.jsonl"
    data.write_text("{}\n")
    monkeypatch.setattr(freeze, "SRC", src)
    monkeypatch.setattr(freeze, "FROZEN", tmp_path / "frozen.json")
    monkeypatch.setattr(freeze, "path_for", lambda s: data)
    rec = freeze.write({"b": 50}, {})
    freeze.check(rec)
    (src / "a.py").write_text("x = 2\n")
    with pytest.raises(freeze.FreezeViolation, match="source"):
        freeze.check(rec)
    (src / "a.py").write_text("x = 1\n")
    data.write_text("{}\n{}\n")
    with pytest.raises(freeze.FreezeViolation, match="data"):
        freeze.check(rec)


def test_rows_match_schema_keys():
    import pathlib

    schema = json.loads((pathlib.Path(__file__).parents[1] / "schemas/forecast_row.v1.json").read_text())
    allowed = set(schema["properties"])
    for split in ("dev", "test"):
        for line in open(pathlib.Path(__file__).parents[1] / f"data/gemma_forecasts_{split}.jsonl"):
            row = json.loads(line)
            assert set(schema["required"]) <= set(row) <= allowed
            assert set(schema["properties"]["forecasts"]["required"]) <= set(row["forecasts"])
