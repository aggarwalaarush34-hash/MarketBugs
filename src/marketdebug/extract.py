"""Extract cached Gemma 4 E4B forecasts from the MarketBugs repo into a compact file.

usage: python -m marketdebug extract <path-to-marketbugs-checkout>

For each bug we keep the gold candidate ids and, for each available note
subset (bitmask over trace, graph, spec, static, semantic), the Gemma
forecast over the 20 candidates averaged in log space over the cached
candidate/note orders. Output: data/gemma_forecasts_{dev,test}.jsonl
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

CIDS = "ABCDEFGHIJKLMNOPQRST"


def logavg(dists):
    n = len(dists[0])
    logs = [sum(math.log(max(d[i], 1e-12)) for d in dists) / len(dists) for i in range(n)]
    m = max(logs)
    e = [math.exp(x - m) for x in logs]
    z = sum(e)
    return [x / z for x in e]


def main(root: str) -> None:
    root = Path(root)
    out_dir = Path(__file__).resolve().parents[2] / "data"
    out_dir.mkdir(exist_ok=True)
    manifest = {"source": "MarketBugs (github.com/raunak2007/marketbugs)", "files": {}}
    for split, run in (("dev", "dev120"), ("test", "heldout_v1")):
        inst = {}
        gold_extra = {}
        for f in (root / "data/marketbugs" / split).glob("*.jsonl"):
            for line in open(f):
                r = json.loads(line)
                if f.name == "eval_only.jsonl":
                    gold_extra[r["id"]] = r
                else:
                    inst[r["id"]] = r
        cands = {}
        for line in open(root / f"artifacts/runs/{run}/test_informed/candidates_inf.jsonl"):
            c = json.loads(line)
            cands[c["id"]] = c
        rows = defaultdict(lambda: defaultdict(list))
        src_values = root / f"artifacts/runs/{run}/test_informed/values_inf.jsonl"
        for src in [src_values, root / f"artifacts/runs/{run}/test_informed/candidates_inf.jsonl",
                    *sorted((root / "data/marketbugs" / split).glob("*.jsonl"))]:
            manifest["files"][str(src.relative_to(root))] = _sha(src)
        revisions = set()
        for line in open(src_values):
            v = json.loads(line)
            probs = v["probs"]
            if isinstance(probs, str):
                probs = json.loads(probs)
            rows[v["id"]][v["subset_mask"]].append(probs)
            revisions.add(v.get("model_revision"))
        n = 0
        with open(out_dir / f"gemma_forecasts_{split}.jsonl", "w") as fh:
            for bid in sorted(rows):
                if bid not in inst or bid not in cands:
                    continue
                gold = set(inst[bid].get("gold_functions") or [])
                pool = cands[bid]["pool"]
                gcids = [c["cid"] for c in pool if f"{c['path']}::{c['qualname']}" in gold]
                k = len(rows[bid][0][0])
                gold_idx = [CIDS.index(c) for c in gcids if CIDS.index(c) < k]
                rec = {
                    "id": bid,
                    "repo": inst[bid]["repo"],
                    "stratum": inst[bid].get("stratum"),
                    "gold": gold_idx,
                    "forecasts": {str(m): logavg(d) for m, d in sorted(rows[bid].items())},
                }
                fh.write(json.dumps(rec) + "\n")
                n += 1
        print(split, n, "bugs")
        manifest[f"{split}_model_revisions"] = sorted(r for r in revisions if r)
        manifest[f"{split}_bugs"] = n
        manifest[f"{split}_output_sha256"] = _sha(out_dir / f"gemma_forecasts_{split}.jsonl")
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")


def _sha(path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main(sys.argv[1])
