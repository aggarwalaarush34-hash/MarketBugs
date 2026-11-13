# DECISIONS.md: decision log

Each entry records what was decided, why, and what it affects. Dates are 2026-11-12 unless noted.

**D1. One exploratory held-out look before the protocol existed.** An early script ran the market on the
held-out split with b = 50, 3 rounds and a 20% budget. It reported top-1 64.3% vs 64.8% for all notes, and
NLL 1.06 vs 1.40. *Why it matters:* the held-out split is not pristine. `PROTOCOL.md` discloses this, and that
script's numbers are not reported as results.

**D2. Rewrote the benchmark to run through the full mechanism.** The exploratory script used only the LMSR
class and edited its internal state directly, bypassing the exchange, evidence board, information rewards and
settlement. The benchmark now runs every bug through `Engine` → `Exchange` → `EvidenceBoard` →
`information.py`, with an audit that the books balance after every run.

**D3. Retrieval misses void the market.** For the 38 held-out bugs whose gold function is not among the 20
candidates, no outcome is true. The exploratory script paid nothing on these markets, so every trader lost its
whole spend. That made trading P&L uniformly negative and drained capital for a reason unrelated to forecast
quality. The markets are now voided and spend refunded. They still count as top-1 errors for every method.
*Alternative considered:* an explicit "not in pool" outcome. Rejected for now because Gemma's forecasts put no
mass on it, so its price would be set by the maker alone.

**D4. Equal-weight log pool is the geometric mean.** The exploratory script multiplied the five forecasts
(weights of 1 each), which is extremely overconfident (NLL 3.87). The baseline now uses weights of 1/5. That is
the standard pooling operator and a fairer comparison.

**D5. Calibration controls added before the frozen run.** The market's exploratory gain was in NLL, and a
budget-limited market partly acts as shrinkage. So the protocol compares it against all-notes with a
dev-fitted temperature and a Hedge pool with dev-fitted temperature. Those are the right controls for a
calibration claim.

**D6. Multi-outcome trading lives in the exchange.** `Exchange.trade_toward` moves prices along the
log-linear path p^(1−α)·belief^α and takes the largest α within budget (bisection). Trades are buy-only. The
binary `trade_to_belief` is the same code with a 2-vector. With unlimited budget α = 1, which maximises a
risk-neutral agent's expected profit under LMSR.

**D7. Information rewards booked, not paid into capital, in the primary run.** Cached agents cannot change
behaviour in response to rewards, so the only channel through which rewards matter here is capital (market
influence). Trading-only capital is the primary configuration. Paying information into capital is an ablation
(E3).

**D8. Held-out Shapley uses the additive approximation.** Held-out bugs have only empty, singleton and full
coalitions. φᵢ = v({i}) + (v(N) − Σⱼ v({j}))/n is exact when every non-additive interaction appears only in
the full coalition. On dev, where the exact value is computable, the approximation's error is large: mean
|error| 0.28 nats against a mean |φ| of 0.35, and the top agent agrees on 79% of bugs. Held-out information
credits are therefore reported as approximate.

**D9. Marginal and pivotal credit cannot run on held-out.** Both need coalitions of 2–4 notes that were
never cached. The ledger ablation reports them as skipped on held-out and runs them on dev.

**D10. Live Gemma agent shipped but not used for results.** `agents/live.py` runs Gemma through a local
Ollama server. The reference machine did not have disk space for the weights, so no reported number uses it.
It is unit-tested with a fake backend. `python -m marketdebug live-smoke` is the entry point.

**D11. Repair ledger evaluated only in simulation.** MarketBugs has no Gemma-written patches. Patch markets,
price-ordered testing and the bounty split are covered by unit tests and a synthetic study
(`bench/repair_sim.py`), labelled SYNTHETIC everywhere.

**D12. Bankruptcy floor of 1 credit.** Without it, an agent driven to zero capital can never trade again, and
one bad early sequence would permanently remove a specialist. Every top-up is journaled (`floor_topups`) and
reported.

**D13. Frozen configuration chosen on 30 dev bugs.** This is small, so the selection may overfit dev. The
one-at-a-time sweeps on held-out (E7, exploratory) show how sensitive the result is to each parameter.
