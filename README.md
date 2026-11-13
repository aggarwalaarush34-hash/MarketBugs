# MarketDebug: Paying Gemma Agents for What They Find

[**Held-out results**](results/RESULTS.md) · [**Protocol**](PROTOCOL.md) · [**Decision log**](DECISIONS.md) · [**Frozen config**](configs/frozen.json)

Multi-agent coding systems usually combine agents by voting, which gives no agent a reason to look for
evidence the others missed. MarketDebug gives each agent capital and positions in LMSR prediction markets about
where a bug is, and pays it through three separate ledgers:

* **Trading**: profit for moving a price toward the truth before others do.
* **Information**: Shapley (or marginal, or pivotal) credit for evidence that improves the group forecast,
  scored against the outcome.
* **Repair**: a bounty for a patch that passes the tests, partly shared with whoever found the evidence it
  cites.

We evaluate the market layer on MarketBugs fault localization, with five Gemma 4 E4B specialists as traders.
The configuration was frozen on 30 development bugs and run once on 210 held-out bugs from seven repositories
the system never saw.

**The short answer is that the market does not find more bugs, and its calibration gain is mostly something
a single temperature does better.** Top-1 is unchanged (inconclusive). The market's log loss is much lower
than a raw all-notes Gemma call. It is higher than the same call with a dev-fitted temperature.

> **What "Gemma" means here.** Every agent belief is a real Gemma 4 E4B-it forecast
> (`google/gemma-4-E4B-it@4ecc2ba`), but they are **cached** outputs produced earlier by the MarketBugs
> pipeline. **No live Gemma inference was run for any number in this repository.** A live Ollama-backed agent
> exists (`agents/live.py`), but it is unit-tested against a fake backend only (see D10 in the decision log).

## Main result

Held-out split, 210 bugs, frozen config (`b = 100`, 1 round, 3.3% of capital per turn, uniform opening,
capital carried from dev):

| | Market (frozen) | All five notes (one Gemma call) | All notes + dev temperature |
|---|---|---|---|
| Top-1, all 210 bugs | 63.3% | 64.8% | 64.8% |
| Gold-function NLL, 172 gold-in-pool bugs | 1.000 | 1.403 | **0.746** |
| ECE (top-label, 10 bins) | 0.251 | 0.307 | 0.180 |
| Paired wins / losses / ties vs all notes | 4 / 7 / 199 | | |

**Primary comparison** (fixed in `PROTOCOL.md`): top-1, market minus all notes = **−1.4 pp**, 95% two-stage
repository-cluster bootstrap CI **[−5.2, +2.9]** (10,000 replicates; instance-level [−4.8, +1.4]). The interval
includes zero, so the result is inconclusive. The market and the single call pick the same function on 189 of
210 bugs.

Other findings:

- **Calibration.** Against the raw all-notes call, the market lowers NLL by 0.40 nats (cluster CI
  [−1.04, −0.01]). Against all notes with a dev temperature, it is 0.25 nats *worse* ([−0.00, +0.53]).
  Against a dev-weighted Hedge pool with temperature, it is 0.25 nats worse ([+0.01, +0.50]). Budget-limited
  trading acts as shrinkage, and explicit temperature scaling shrinks better.
- **Shapley credit works where it can be computed exactly.** On dev, with all 32 note coalitions cached,
  Shapley payouts add up to the total value to 4e-15 and rank the graph specialist far above the rest (0.17
  nats/bug). Marginal credit is unstable: the top-credited agent changes with reveal order in 67% of orders.
  Pivotal credit overpays on 46% of bugs.
- **Held-out information credit is approximate.** Only singleton and full coalitions were cached there, so
  held-out Shapley uses an additive approximation. On dev its mean error is 0.28 nats and it picks the same
  top agent on 79% of bugs.
- **The copycat rule works.** A copycat of the graph specialist would take 0.11 nats/bug of Shapley credit
  without the evidence board's dedupe rule. With the rule it earns zero on every bug.
- **Paying information into capital does not help on held-out** (NLL +0.03, CI [+0.01, +0.05]).
- **Conditioning on public notes is mixed on dev.** When agents can read already-revealed notes (possible only
  on dev, where every coalition forecast exists), dev top-1 rises from 73.3% to 76.7% (one bug), but NLL gets
  worse (0.915 → 0.983) under the frozen config. Under the unfrozen default config, top-1 rose from 60.0% to
  76.7%. This is exploratory, on 30 bugs, and could not be tested on held-out.
- **Repair ledger (SYNTHETIC simulation, 500 worlds).** Testing patches in market-price order needs 1.29 test
  runs per solved bug, against 1.45 for random order and 1.55 for proposal order. Solve rates within the
  3-test budget are similar (91% / 92% / 82%).

Every table, including per-repository and per-stratum results, robustness sweeps and ledgers, is in
[`results/RESULTS.md`](results/RESULTS.md). The raw output is [`results/heldout.json`](results/heldout.json).

## How it works

Each MarketBugs bug comes with 20 retrieved candidate functions and five specialist notes. Each note gives
Gemma a different view of the bug: trace, graph, spec, static or semantic. One bug is one episode:

1. **Open a market.** A 20-outcome LMSR market over the candidates (`lmsr.py`), funded by the operator. With
   uniform opening the maker's worst-case loss is `b ln 20` per market. In general it is `b ln(1/min p0)`.
2. **Trade.** Each specialist's belief is Gemma's forecast given its own note (`agents/cached.py`). On its
   turn it moves prices along `p^(1−α) · belief^α`, taking the largest `α` its per-turn budget allows
   (`Exchange.trade_toward`, buy-only). With unlimited budget `α = 1`, the profit-maximising trade under LMSR.
3. **Reveal.** Each specialist posts its note to a timestamped evidence board (`evidence.py`). A fact counts
   for credit only if it was not already public.
4. **Settle.** The market resolves with payout split evenly over the gold candidates. If retrieval missed
   the gold function, it is voided and spend refunded. Trading P&L moves capital.
5. **Information rewards.** `v(S) = log P_S(gold) − log P_∅(gold)`, where `P_S` is Gemma's forecast given the
   notes in `S` (`information.py`). Exact Shapley where every coalition was evaluated, the additive
   approximation otherwise. Marginal and pivotal credit are also available. Rewards are booked to the
   information ledger and optionally paid into capital.
6. **Audit.** Every credit movement is journaled, and `Exchange.audit()` checks after every run that each
   account's capital is explained by the journal.

Capital lives in one exchange across the bugs of a run, so capital earned on earlier bugs becomes market
influence on later ones. For frozen agents, this capital channel is the only one through which rewards can
matter.

The **repair ledger** (`settlement.py`) gives every proposed patch a "will this pass?" market. It tests patches
in descending price order up to a test budget, accepts the first that passes, and resolves root-cause markets
from the accepted patch's diff. It pays a bounty to the author, 30% of which is shared among the original
finders of the facts it cites. MarketBugs has no Gemma-written patches, so this part is covered by unit tests
and a synthetic study (`bench/repair_sim.py`).

The **agent interface** (`protocol.py`) is the same for cached, live and simulated agents. Agents see an
`Observation` (prices, positions, capital, public evidence, patches) and return actions: `Reveal`, `Trade`,
`TradeDistribution`, `OpenLocationMarket` and `ProposePatch`. `engine.py` runs rounds and applies them.

## Baselines

All baselines use the same cached Gemma forecasts as the market (`bench/baselines.py`). Fitted parameters come
from dev only.

| Name | What it is |
|---|---|
| `no_notes`, `single.<agent>` | Gemma with no notes, or with one specialist's note |
| `all_notes` | One Gemma call with all five notes, averaged over 5 note orders (MarketBugs' B_full5) |
| `all_notes.T` | The same, with a scalar temperature fitted on dev (T = 2.41) |
| `linpool`, `logpool` | Equal-weight linear and geometric pools of the five single-note forecasts |
| `hedge`, `hedge.T` | Log pool with Hedge weights learned on dev (η = 0.1 by 2-fold CV), optionally tempered |

## Data

`data/gemma_forecasts_{dev,test}.jsonl` holds one row per bug: the gold candidate indices and, per note
subset (bitmask over trace, graph, spec, static, semantic), Gemma's distribution over the 20 candidates. The
schema is `schemas/forecast_row.v1.json`.

| split | repositories | bugs | gold in pool | coalitions cached |
|---|---|---|---|---|
| dev | cachetools, intervaltree, sortedcontainers | 30 | 28 | all 32 |
| held-out | attrs, boltons, dateutil, isodate, more-itertools, tomli, toolz | 210 | 172 | ∅, 5 singletons, full |

The splits are repository-disjoint. The data is extracted from the
[MarketBugs](https://github.com/raunak2007/marketbugs) artifacts with `python -m marketdebug extract
<checkout>`, and the sha256 of every source file used is recorded in `data/MANIFEST.json`.

## Reproducing the results

Everything runs on a CPU, with no GPU or network needed.

```bash
uv sync --extra figures --extra live   # or: make setup
make test                              # 61 unit tests
make validate                          # validate both data splits
make dev                               # dev-only experiments (~2 s)
make freeze                            # select the config on dev, write configs/frozen.json (~25 s)
make heldout                           # every held-out experiment from the frozen config (~30 min)
```

`heldout` refuses to run if any source file or data file has changed since `freeze` (sha256 contract in
`configs/frozen.json`). CI checks the contract on every push. Most of the 30 minutes goes to the 10,000-replicate
bootstrap and the 500-world repair simulation.

To try a live Gemma agent (not used for any reported number):

```bash
ollama pull gemma4:e4b
python -m marketdebug live-smoke --model gemma4:e4b
```

## Repository layout

```
src/marketdebug/
  lmsr.py          n-outcome LMSR market maker (pricing, buy-only vectors, weighted resolution, void)
  exchange.py      accounts, three ledgers, multi-outcome trading, bankruptcy floor, journal + audit
  evidence.py      timestamped evidence board with copycat dedupe
  information.py   scoring rules; marginal, Shapley (exact / sampled / additive approx), pivotal credit
  settlement.py    patch markets, price-ordered testing, root-cause resolution, repair bounty
  protocol.py      Observation and actions shared by every agent type
  engine.py        episode runner
  agents/          cached Gemma agents (+ copycat), live Ollama Gemma agent
  bench/           localization benchmark, baselines, SYNTHETIC repair simulation
  analysis/        metrics (top-k, NLL, Brier, ECE, temperature), bootstrap, experiments, report
  data.py          loading and validation of cached forecasts
  extract.py       rebuild data/ from a MarketBugs checkout
  freeze.py        frozen evaluation contract
tests/             unit and integration tests (mechanism, data, benchmark, freeze, live agent, sim)
configs/           frozen.json
data/              cached Gemma forecasts + provenance manifest
results/           RESULTS.md, heldout.json, dev_selection.json, dev_only.json
schemas/           JSON schema for forecast rows
PROTOCOL.md        analysis plan and disclosure of the one prior held-out look
DECISIONS.md       decision log
```

## Limitations

- **No live Gemma inference.** Beliefs are cached Gemma forecasts, and the live agent is only mock-tested.
- **Frozen agents.** The specialists cannot respond to rewards. The only incentive channel tested is capital,
  meaning market influence. Whether Gemma agents that see their balances in context, or are fine-tuned on
  these rewards, investigate differently is untested.
- **Not independent.** All five specialists are the same small model, so they agree on the top candidate
  for most bugs, which limits any aggregation method.
- **Small dev split.** 30 bugs chose the frozen configuration. The held-out sweeps show NLL is sensitive to
  the budget fraction (0.86 at 1% per turn vs 1.39 at 30%).
- **One prior look at held-out.** An exploratory run looked at held-out once before this protocol existed
  (disclosed in `PROTOCOL.md` and D1).
- **Wide intervals.** Seven held-out repositories give seven bootstrap clusters.
- **Approximate held-out credit.** Held-out information credit uses an approximation with substantial error.
  Marginal and pivotal credit could not run on held-out.
- **Synthetic repair results.** The repair ledger is evaluated only in simulation. Mutants are not natural bugs.

## Citation

```bibtex
@misc{marketdebug2026,
  title  = {MarketDebug: Paying Gemma Agents for What They Find},
  author = {Aggarwal, Aarush and Chakraborty, Ratul and Mondal, Raunak and Raheja, Aarush},
  year   = {2026},
  note   = {Kaggle Gemma 4 Developer Agent Paper Track}
}
```

This builds on MarketBugs (Mondal et al., 2026), which supplied the benchmark and the cached Gemma forecasts.
