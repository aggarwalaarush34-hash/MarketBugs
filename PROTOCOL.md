# PROTOCOL.md: analysis plan and evaluation contract

This file fixes what counts as the result before the frozen held-out run.

**Disclosure.** It is not a preregistration in the strict sense. On 2026-11-12, before this protocol and the
current code existed, an exploratory script (since deleted) was run once on the held-out split with an
unfrozen configuration (b = 50, 3 rounds, 20% budget). That run is recorded in `DECISIONS.md` (D1). It used
code that was later found to mis-settle retrieval misses (D3). Its numbers are not reported as results. After it:

* every design choice and the market configuration were made using the **development split only**
  (`python -m marketdebug freeze`), and
* the held-out split was evaluated once with the frozen configuration (`python -m marketdebug heldout`),
  enforced by `configs/frozen.json` (sha256 of both data files and of every source file).

Readers should still treat the held-out numbers as having had one prior look.

## Data

* **Source.** Cached Gemma 4 E4B-it forecasts from MarketBugs (`github.com/raunak2007/marketbugs`), model
  revision `google/gemma-4-E4B-it@4ecc2ba63bcac23ba4a767fc87025b1450f84c9c`. Rebuilt by
  `python -m marketdebug extract <checkout>`. Provenance hashes are in `data/MANIFEST.json`.
* **Dev.** 30 bugs (cachetools, intervaltree, sortedcontainers; 10 each). All 32 note subsets are cached,
  and 28 have the gold function in the pool.
* **Held-out.** 210 bugs from 7 repositories never used for any decision (attrs, boltons, dateutil, isodate,
  more-itertools, tomli, toolz; 30 each). Only the empty, five singleton and full note subsets are cached,
  and 172 have the gold function in the pool.
* **Forecasts.** Per subset, the Gemma probability agent's distribution over the 20 candidates, averaged in
  log space over the cached orders (2 candidate orders; the all-notes forecast also over 5 note orders).

## Primary (confirmatory) comparison, the only one

* **Comparison:** top-1 accuracy of `market` minus top-1 of `all_notes`, held-out split, all 210 bugs.
  A retrieval miss counts as an error for both.
  * `market`: the frozen configuration, with the five cached-Gemma specialists as traders and capital carried
    from the dev sequence into the held-out sequence (one Exchange).
  * `all_notes`: one Gemma call that sees all five notes, the strongest prior baseline (MarketBugs' B_full5).
* **Statistic:** mean paired per-bug difference.
* **Inference:** two-stage repository-cluster bootstrap (resample repositories, then bugs within them),
  10,000 replicates, seed 20261112. The instance-level interval is reported beside it.
* **Reading:** if the 95% cluster interval contains 0, the result is inconclusive. That does not show
  equivalence.

## Secondary (exploratory: effect sizes and intervals, no significance claims)

1. Gold-function NLL, Brier, ECE (top-label, 10 bins), top-3, top-5 and MRR for every method.
2. Calibration controls: `market` vs `all_notes.T` (dev-fitted temperature) and vs `hedge.T` (Hedge log pool
   with dev-learned weights and temperature). The market's calibration gain is only interesting if it survives
   these.
3. Ledger ablation: capital moves with trading only vs trading + information (Shapley / marginal / pivotal).
4. Credit properties on dev (exact, all 32 coalitions): Shapley efficiency, pivotal budget gap, marginal
   order dependence, and the error of the additive Shapley approximation used on held-out.
5. Copycat: information credit with and without the evidence board's dedupe rule.
6. Reveal policy (trade-then-reveal, reveal-first, hoard) and conditioning on public notes (dev only).
7. Robustness: 20 turn-order seeds, 20 bug-order seeds, one-at-a-time parameter sweeps.
8. Repair ledger: **synthetic** simulation only (no Gemma patches exist), comparing price-ordered vs proposal-
   order vs random-order testing.
9. Per-repository and per-stratum tables for every headline method.

## Fixed constants

* Freeze grid (dev only): b ∈ {10, 25, 50, 100, 200}; per-turn budget fraction ∈ {0.0333, 0.0667, 0.15};
  rounds ∈ {1, 3, 5}; opening prices ∈ {no-notes forecast, uniform}. Objective: dev gold NLL of the
  carried-capital market. Ties go to the earlier grid entry.
* Starting capital 1,000 credits per specialist, bankruptcy floor 1 credit, information scale 100 credits per
  nat, log scoring rule, capital moves with trading P&L only (information is booked but not paid into capital
  in the primary run).
* Temperatures: one scalar per method, fitted on dev by NLL. Hedge η ∈ {0.1, 0.3, 1, 3} by 2-fold CV on dev.
* Ties at the top of a forecast break to the lowest candidate index.
* Information value: v(S) = log P_S(gold) − log P_∅(gold), where P_S is the cached Gemma forecast given the
  notes in S. Exact Shapley where all coalitions exist (dev), additive approximation otherwise (held-out).

## Reporting rules

* Report gold-in-pool and unconditional results.
* Report cluster and instance intervals side by side.
* Give every exclusion with its count and reason. Retrieval misses are voided markets, not exclusions.
* Report null and negative results in the same tables as positive ones.
* Label information credit as "value of a note given the frozen Gemma probability agent". It is not a
  claim about agents' real incentives: the cached agents do not respond to rewards.
* Label repair-ledger numbers SYNTHETIC wherever they appear.
