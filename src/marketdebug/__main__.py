"""Command line: python -m marketdebug <command>

  extract <marketbugs-checkout>   rebuild data/ from MarketBugs artifacts
  validate                        validate both data splits
  freeze                          select the market config on dev, write configs/frozen.json
  heldout [--reps N]              run every experiment with the frozen config
  repair-sim [--n N]              synthetic repair-ledger study
  dev [--reps N]                  run every experiment on dev only (no held-out access)
  live-smoke [--model M]          one toy episode with LiveGemmaAgent against a local Ollama
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from .data import ROOT, load_split, validate, path_for, parse


def _dump(obj, path):
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


def cmd_validate(_):
    ok = True
    for s in ("dev", "test"):
        bugs = [parse(json.loads(l)) for l in open(path_for(s))]
        probs = validate(bugs)
        full = sum(b.has_full_table for b in bugs)
        gip = sum(b.gold_in_pool for b in bugs)
        print(f"{s}: {len(bugs)} bugs, {gip} gold in pool, {full} with all 32 coalitions, {len(probs)} problems")
        for p in probs[:10]:
            print("  ", p)
        ok &= not probs
    return 0 if ok else 1


def cmd_freeze(_):
    from .analysis.experiments import select_on_dev
    from . import freeze

    dev = load_split("dev")
    t = time.time()
    sel = select_on_dev(dev)
    best = sel["selected"]
    print(f"selected on dev in {time.time() - t:.0f}s: {best['config']}  dev NLL {best['dev']['nll']:.4f} top1 {best['dev']['top1']:.3f}")
    _dump(sel, ROOT / "results" / "dev_selection.json")
    rec = freeze.write(best["config"], {"objective": sel["objective"], "dev": best["dev"], "n_trials": len(sel["trials"])})
    print(f"froze config, code {rec['code_sha256'][:12]}")
    return 0


def cmd_heldout(a):
    from . import freeze
    from .analysis.experiments import all_experiments
    from .analysis.report import render
    from .bench.localization import MarketConfig
    from .bench.repair_sim import study

    rec = freeze.check()
    cfg = MarketConfig.from_dict(rec["config"])
    dev, test = load_split("dev"), load_split("test")
    t = time.time()
    res = all_experiments(dev, test, cfg, reps=a.reps)
    res["repair_sim"] = study(n=a.sim_n)
    res["frozen"] = rec
    res["runtime_s"] = time.time() - t
    _dump(res, ROOT / "results" / "heldout.json")
    (ROOT / "results" / "RESULTS.md").write_text(render(res))
    print("wrote results/RESULTS.md")
    p = res["main"]["primary"]["cluster"]
    print(f"PRIMARY top-1 market - all_notes: {100*p['mean']:+.2f} pp [{100*p['lo']:+.2f}, {100*p['hi']:+.2f}]")
    return 0


def cmd_dev(a):
    from .analysis.experiments import credit_properties, copycat, conditioning
    from .bench.localization import MarketConfig

    dev = load_split("dev")
    cfg = MarketConfig()
    res = {"credit_properties": credit_properties(dev), "copycat": copycat(dev, cfg), "conditioning": conditioning(dev, cfg)}
    _dump(res, ROOT / "results" / "dev_only.json")
    return 0


def cmd_repair(a):
    from .bench.repair_sim import study

    _dump(study(n=a.n), ROOT / "results" / "repair_sim.json")
    return 0


def cmd_extract(a):
    from .extract import main

    main(a.root)
    return cmd_validate(a)


def cmd_live(a):
    from .agents.live import LiveGemmaAgent, OllamaBackend
    from .bench.localization import new_exchange, MarketConfig
    from .engine import Engine, EngineConfig
    from .lmsr import LMSRMarket

    task = {
        "issue": "parse_int('007') raises ValueError; leading zeros should be accepted.",
        "candidates": [{"path": "num.py", "qualname": n} for n in ("parse_int", "format_int", "is_digit", "strip_sign")],
    }
    notes = {"trace": "Traceback ends in parse_int at the regex match.", "spec": "Docstring of parse_int allows leading zeros."}
    ex = new_exchange(MarketConfig())
    m = LMSRMarket.opening_at("bug:toy", "which?", [0.25] * 4, 50)
    m.meta.update(kind="localization")
    ex.add_market(m)
    be = OllamaBackend(model=a.model)
    agents = [LiveGemmaAgent(k, k, v, be) for k, v in notes.items()]
    Engine(agents, cfg=EngineConfig(rounds=2), exchange=ex, task=task).play()
    print("prices:", [round(x, 3) for x in m.prices()])
    for ag in agents:
        print(ag.name, "parsed replies:", sum(r["parsed"] for r in ag.replies), "/", len(ag.replies))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="marketdebug")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    sub.add_parser("freeze")
    h = sub.add_parser("heldout")
    h.add_argument("--reps", type=int, default=10000)
    h.add_argument("--sim-n", type=int, default=500)
    d = sub.add_parser("dev")
    d.add_argument("--reps", type=int, default=2000)
    r = sub.add_parser("repair-sim")
    r.add_argument("--n", type=int, default=500)
    e = sub.add_parser("extract")
    e.add_argument("root")
    lv = sub.add_parser("live-smoke")
    lv.add_argument("--model", default="gemma4:e4b")
    a = ap.parse_args(argv)
    return {
        "validate": cmd_validate, "freeze": cmd_freeze, "heldout": cmd_heldout, "dev": cmd_dev,
        "repair-sim": cmd_repair, "extract": cmd_extract, "live-smoke": cmd_live,
    }[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
