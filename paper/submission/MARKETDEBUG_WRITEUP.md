# MarketDebug: Paying Gemma Agents for What They Find

**Prediction markets, information rewards and repair bounties for small-model fault localization**

Aarush Aggarwal, Ratul Chakraborty, Raunak Mondal, Aarush Raheja

## Abstract

Multi-agent coding systems usually combine agents by voting or by concatenating their outputs. Neither gives an agent a reason to search for evidence the others missed. We present MarketDebug, a mechanism that gives each agent capital and positions in logarithmic market scoring rule (LMSR) prediction markets about where a bug is. Rewards come from three separate ledgers: trading profit for being right early, Shapley information rewards for evidence that improves the group forecast, and a repair bounty for a passing patch. We evaluate the market layer on MarketBugs, a held-out set of 210 single-function bugs from seven Python repositories. Five Gemma 4 E4B specialists act as traders, and their beliefs are frozen Gemma forecasts. The market matches the strongest baseline on top-1 localization (64.3% vs. 64.8%; paired difference $-0.5$ pp, 95% repository-cluster CI $[-4.8, +3.8]$). It reduces log loss on the gold function from 1.40 to 1.06 nats. Exact Shapley credit identifies the specialists whose evidence matters. Markets are therefore a practical way to obtain calibrated forecasts from small local models, but they do not by themselves find more bugs.

## Introduction

Small open models that run on one accelerator are weak at multi-step software engineering. A common response is to run several instances and aggregate them. When every instance is the same small model, aggregation has little independent signal to work with. Prior work on MarketBugs found exactly this: a learned pool of five Gemma specialists was statistically indistinguishable from a single call that saw all five notes \[1\].

Aggregation without incentives is a poll. We ask a different question: *what if each agent were paid for information the group does not yet have?* Markets do this for people. A trader profits only by disagreeing with the price and being right. Conitzer showed how scoring rules and cooperative game theory can reward the contribution of information itself \[2\], and Hanson’s LMSR provides liquidity even with few traders \[3\].

Our contributions:

- **A three-ledger incentive design** for debugging agents. It separates trading rewards, information rewards and repair rewards, so that no single behavior can be gamed for all three (Section 3).

- **An open implementation**: LMSR markets, an evidence board that refuses credit to copied facts, marginal, Shapley and pivotal information rewards, and test-ordered patch settlement, with a common agent interface for simulated or Gemma agents.

- **An evaluation on real Gemma 4 E4B forecasts** over 240 MarketBugs bugs. The market improves calibration substantially without changing accuracy, and exact Shapley values rank the specialists consistently with their standalone usefulness (Section 4).

![Figure 1](figures/fig1_overview.png)

*Figure 1. **MarketDebug.** Top: agents trade on private beliefs and reveal evidence; one settlement pays three separate ledgers. Bottom: on 210 held-out MarketBugs bugs, the five Gemma specialists trading in an LMSR market produce markedly better-calibrated forecasts than any single Gemma call, including one that sees all notes. Top-1 accuracy is unchanged (Table 1, Figure 2).*

## Related work

**Agentic software engineering.** SWE-bench \[6\] and SWE-agent \[7\] established repository-level repair with large API models. Agentless \[8\] showed that a fixed localize-then-repair pipeline is competitive. Fault localization remains the main bottleneck for small models \[1\]. **Multi-agent aggregation.** Self-consistency \[9\] and multi-agent debate \[10\] combine samples from the same model. They pay no attention to which agent contributed what. **Markets and information elicitation.** Proper scoring rules \[5\] reward honest probabilities. The LMSR \[3\] turns scoring rules into an automated market maker with bounded subsidy $b\ln n$. Conitzer \[2\] connects prediction markets, marginal information rewards and the Shapley value \[4\]. To our knowledge this is the first application of these mechanisms to coding agents.

## Method

**Markets.** For each bug we open an $n$-outcome LMSR market over the retrieved candidate functions. The cost function is $C(\mathbf{q}) = b \ln \sum_i e^{q_i/b}$, and the price of outcome $i$ is $\mathrm{softmax}(\mathbf{q}/b)_i$. A share of outcome $i$ pays one credit if $i$ is the buggy function. The market opens at the no-notes Gemma forecast.

**Agents and trading.** Each agent holds capital (1,000 credits) and a belief $\mathbf{p}_a$. On its turn it moves the log-prices a fraction $\alpha\in[0,1]$ of the way toward its belief, $\mathbf{t}\propto \mathbf{p}^{1-\alpha}\mathbf{p}_a^{\alpha}$. It chooses the largest $\alpha$ whose cost stays within its per-turn budget (20% of capital, split over three rounds). Shares are buy-only. A risk-neutral agent with unlimited capital would set $\alpha=1$. The budget turns the market into a capital-weighted logarithmic pool whose weights are *earned*.

**Ledger 1: trading.** At settlement each agent receives its shares on the gold outcome. Trading P&L is payout minus spend. Capital carries to the next bug, so agents that were right gain influence. This is the selection channel through which rewards change the behavior of a frozen LLM.

**Ledger 2: information.** Each revealed fact is posted to a timestamped board. A fact counts only if it was not already public, which prevents a copycat from splitting the original finder’s credit. Let $v(S) = s(\mathbf{p}_S, x) - s(\mathbf{p}_\emptyset, x)$ for a proper scoring rule $s$ (log score here). Here $\mathbf{p}_S$ is the forecaster’s output given only the facts of coalition $S$, and $x$ is the realized outcome. We implement three credit rules: marginal (score change in filing order), Shapley $\phi_i = \frac{1}{n!}\sum_\pi [v(P_i^\pi\cup\{i\}) - v(P_i^\pi)]$ (exact for $n\le 8$, permutation-sampled otherwise), and pivotal $v(N)-v(N\setminus\{i\})$. Shapley credit is budget-balanced and independent of reveal order. Pivotal credit generally is not. These are Conitzer-inspired. Their incentive guarantees rest on assumptions that LLM agents need not satisfy, so we treat them as hypotheses to test.

**Ledger 3: repair.** Every proposed patch gets a binary market “will this patch pass?”. Patches are tested in descending price order up to a test budget, and the first passing patch is accepted. Its author receives a bounty, 30% of which is shared among the original finders of the facts it cites. Root-cause markets resolve from the accepted patch’s diff, which gives diagnostic questions a checkable answer.

## Experiments

**Data.** MarketBugs \[1\] contains single-function AST mutants that fail a test deterministically. We use the 30 development bugs with complete coalition tables (all 32 note subsets) and the 210 held-out bugs from seven repositories never used in development (attrs, boltons, dateutil, isodate, more-itertools, tomli, toolz). Each bug has 20 retrieved candidates. The gold function is in the pool for 172 of the 210 held-out bugs, and retrieval misses count as errors for every method.

**Agents.** The five specialists (trace, graph, spec, static, semantic) each write a note from a different view of the bug. Beliefs are the frozen Gemma 4 E4B-it probability-agent forecasts given only that specialist’s note. They are read from the logits of 20 single-token labels and averaged in log space over two candidate orders. All forecasts are cached outputs of the pinned model revision `google/gemma-4-E4B-it@4ecc2ba` \[1\]. The market layer runs on a CPU in seconds.

**Methods.** *No notes*; each *single* specialist; *all notes* (one Gemma call with all five notes, averaged over five note orders, the strongest prior baseline); an equal-weight *log pool*; the *market* with fresh equal capital per bug; and the *market* with capital carried across bugs (dev first, then held-out). Market settings: $b=50$, 3 rounds, random turn order, seed 0.

| Method                                   | Top-1 (%) $\uparrow$ | Log loss (nats) $\downarrow$ |
|:-----------------------------------------|:--------------------:|:----------------------------:|
| No notes                                 |         62.4         |            1.537             |
| Single: trace / graph / spec             |  61.4 / 63.8 / 63.8  |    1.449 / 1.334 / 1.472     |
| Single: static / semantic                |     66.2 / 63.3      |        1.298 / 1.450         |
| All five notes (one call)                |         64.8         |            1.403             |
| Log pool, equal weights                  |         64.8         |            3.872             |
| Market, fresh capital                    |         63.3         |            1.138             |
| Market, capital from dev                 |         64.3         |          **1.058**           |
| Market, carried capital from equal start |         63.8         |            1.061             |

*Table 1. Held-out results (210 bugs). Top-1 counts retrieval misses as errors. Log loss is on the gold function for the 172 bugs with gold in the pool.*

**Accuracy is unchanged.** The paired top-1 difference between the market with carried capital and the all-notes call is $-0.5$ pp. Its 95% two-stage repository-cluster bootstrap CI is $[-4.8, +3.8]$ (10,000 replicates). The fresh-capital market gives $-1.4$ pp $[-5.2, +2.4]$. Like the prior pooling result \[1\], specialists built from one small model mostly agree on the top candidate, so no aggregation rule moves accuracy much. Figure 2 shows all three contrasts.

![Figure 2](figures/fig2_paired.png)

*Figure 2. Paired top-1 difference against the all-notes call on the 210 held-out bugs, with 95% two-stage repository-cluster bootstrap intervals. Every interval spans zero.*

**Calibration improves substantially.** Log loss falls from 1.403 (all notes) and 1.298 (best single) to 1.058 nats (Table 1, Figure 1). The equal-weight log pool, which multiplies five overconfident forecasts, is badly overconfident (3.87 nats). The market avoids this because budgets bound how far any trader can move the price. On the 30 dev bugs the pattern is the same: the fresh-capital market matches all notes on top-1 (76.7%) and lowers log loss from 1.379 to 1.198.

**Shapley credit tracks usefulness.** On dev, where all 32 coalitions were cached, exact Shapley values in nats per bug rank graph (0.173) far above semantic (0.066), static (0.053), trace (0.050) and spec (0.027). On held-out bugs only singletons and the full set are cached. There we use the approximation $\phi_i = v(\{i\}) + (v(N) - \sum_j v(\{j\}))/n$, which is exact when interactions appear only in the full coalition. It gives static (0.129) and graph (0.093) positive credit and the other three slightly negative credit. This agrees with Table 1, where static and graph are the best single specialists. Capital after the held-out run tells the same story: graph 535, static 310, trace 157, spec 136, semantic 10 credits. Figure 3 shows the credit and the capital paths.

![Figure 3](figures/fig3_credit.png)

*Figure 3. Left: mean Shapley information credit per specialist, exact on the 30 dev bugs and approximate on the 210 held-out bugs. Right: each trader's capital as it carries across the dev bugs and then the held-out bugs.*

**Every trader loses money.** Held-out trading P&L is negative for all five agents ($-376$ to $-999$ credits). Gemma’s single-note forecasts are overconfident, so agents overpay for their favourite candidates, and the market maker’s bounded subsidy does not cover the loss. Trading P&L alone is therefore a poor reward for these agents. The separate information ledger is what credits the useful specialists, which supports keeping the ledgers apart.

## Discussion and limitations

The market layer is cheap and model-agnostic, and it gives better-calibrated localization forecasts. That matters for an agent that must decide which candidate to patch first, or when to stop and run tests. It does not increase accuracy here, because the specialists share one base model and are not independent. Several limitations apply. Our traders are frozen forecasts, not agents that choose what to investigate. The incentive channel tested is capital selection, not in-context or RL-trained response to rewards. Only seven held-out repositories bound the bootstrap. Held-out Shapley values use an approximation. Mutants are not natural bugs. The repair ledger and patch markets are implemented but were not run end-to-end with Gemma-generated patches before the deadline. The natural next steps are diverse specialists (different models, tools and starting files), agents that observe prices and balances in context, and RL fine-tuning of Gemma on the information-reward signal.

## Conclusion

Paying agents for being right, for useful evidence, and for working fixes separately is easy to implement and auditable. On real Gemma 4 E4B forecasts it produces a calibrated group forecast and a credit assignment that identifies the useful specialists. Accuracy gains will require agents whose information actually differs.

**Reproducibility.** Code, cached forecasts and the benchmark script (`python -m marketdebug.gemma_bench`) are in the project repository. The forecasts come from the public MarketBugs artifacts \[1\].

## References

1. R. Mondal et al. MarketBugs: What makes evidence aggregation reliable for small-model software debugging? <https://github.com/raunak2007/marketbugs>, 2026.
2. V. Conitzer. Prediction markets, mechanism design, and cooperative game theory. In *UAI*, 2009.
3. R. Hanson. Combinatorial information market design. *Information Systems Frontiers*, 5(1):107–119, 2003.
4. L. S. Shapley. A value for $n$-person games. In *Contributions to the Theory of Games II*, 1953.
5. T. Gneiting and A. E. Raftery. Strictly proper scoring rules, prediction, and estimation. *JASA*, 102(477):359–378, 2007.
6. C. E. Jimenez et al. SWE-bench: Can language models resolve real-world GitHub issues? In *ICLR*, 2024.
7. J. Yang et al. SWE-agent: Agent-computer interfaces enable automated software engineering. In *NeurIPS*, 2024.
8. C. S. Xia et al. Agentless: Demystifying LLM-based software engineering agents. *arXiv:2407.01489*, 2024.
9. X. Wang et al. Self-consistency improves chain of thought reasoning in language models. In *ICLR*, 2023.
10. Y. Du et al. Improving factuality and reasoning in language models through multiagent debate. In *ICML*, 2024.
