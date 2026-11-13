from marketdebug.agents.live import LiveGemmaAgent, build_prompt, parse_probs
from marketdebug.bench.localization import MarketConfig, new_exchange
from marketdebug.bench.repair_sim import SimConfig, run_once, study
from marketdebug.engine import Engine, EngineConfig
from marketdebug.lmsr import LMSRMarket


class FakeBackend:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else "nonsense"


def test_parse_probs():
    p = parse_probs('Sure! {"probs": {"A": 3, "B": 1}}', 3)
    assert len(p) == 3 and abs(sum(p) - 1) < 1e-9 and p[0] > p[1] > p[2] > 0
    assert parse_probs("no json here", 3) is None
    assert parse_probs('{"probs": {"A": -1}}', 2) is None
    assert parse_probs('{"probs": {"A": "lots"}}', 2) is None
    assert parse_probs('{"probs": {"A": 0, "B": 0}}', 2) is None
    assert parse_probs('{"A": 1, "B": 1}', 2) is not None  # bare mapping accepted


def episode(backend):
    ex = new_exchange(MarketConfig())
    m = LMSRMarket.opening_at("bug:toy", "q", [0.25] * 4, 50)
    m.meta["kind"] = "localization"
    ex.add_market(m)
    task = {"issue": "boom", "candidates": [{"path": "a.py", "qualname": f"f{i}"} for i in range(4)]}
    ag = LiveGemmaAgent("trace", "trace", "points at f2", backend)
    Engine([ag], cfg=EngineConfig(rounds=2), exchange=ex, task=task).play()
    return m, ag


def test_live_agent_trades_on_parsed_reply():
    be = FakeBackend(['{"probs": {"A": 0.05, "B": 0.05, "C": 0.85, "D": 0.05}}'] * 2)
    m, ag = episode(be)
    assert m.price(2) > 0.25
    assert all(r["parsed"] for r in ag.replies)
    assert "points at f2" in be.prompts[0] and "C. a.py::f2" in be.prompts[0]


def test_live_agent_abstains_on_garbage():
    m, ag = episode(FakeBackend(["I think it's C"] * 2))
    assert m.prices() == [0.25] * 4 or max(abs(p - 0.25) for p in m.prices()) < 1e-12
    assert not any(r["parsed"] for r in ag.replies)


def test_prompt_lists_public_notes():
    p = build_prompt({"issue": "x", "candidates": [{"path": "p", "qualname": "q"}] * 2}, "spec", "mine", [("graph", "theirs")], 2)
    assert "[graph] theirs" in p and "A-B" in p


def test_repair_sim_deterministic_and_sane():
    cfg = SimConfig(n_agents=5, n_locations=4)
    a = run_once(cfg, 11, "price")
    b = run_once(cfg, 11, "price")
    assert a == b
    assert a.tests_used <= cfg.test_budget
    if a.solved:
        assert abs(sum(a.repair.values()) - 200) < 1e-6 and a.author_correct
    s = study(cfg, n=20)
    assert s["synthetic"] is True
    for o in ("price", "proposal", "random"):
        assert 0 <= s[o]["solve_rate"] <= 1
