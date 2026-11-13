"""Accounts, the three reward ledgers, and an exchange that hosts LMSR markets.

Trading moves capital: credits are debited when shares are bought and
credited at settlement. Information and repair rewards are booked to their
own ledgers so each behaviour can be measured on its own. Whether a ledger
also flows into capital (and so into an agent's future market influence) is
a run-level choice: `Exchange.capital_from` lists the ledgers that do.

Every movement of credits is written to `Exchange.journal`, so a run can be
audited after the fact and the books checked to balance (see `audit`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .lmsr import LMSRMarket, Trade, geometric_mix

LEDGERS = ("trading", "information", "repair")


class InsufficientFunds(Exception):
    pass


@dataclass
class Account:
    agent: str
    capital: float
    trading: float = 0.0
    information: float = 0.0
    repair: float = 0.0
    floor_topups: float = 0.0   # credits injected by the bankruptcy floor

    @property
    def total_reward(self) -> float:
        return self.trading + self.information + self.repair


@dataclass
class Exchange:
    starting_capital: float = 1000.0
    default_b: float = 100.0
    max_markets: int = 1000
    max_fraction_per_trade: float = 0.25
    capital_floor: float = 0.0
    capital_from: tuple[str, ...] = ("trading",)
    accounts: dict[str, Account] = field(default_factory=dict)
    markets: dict[str, LMSRMarket] = field(default_factory=dict)
    journal: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        bad = set(self.capital_from) - set(LEDGERS)
        if bad:
            raise ValueError(f"unknown ledgers {bad}")
        if "trading" not in self.capital_from:
            raise ValueError("trading P&L always moves capital")

    # ---------- accounts ----------
    def register(self, agent: str, capital: float | None = None) -> Account:
        if agent not in self.accounts:
            c = self.starting_capital if capital is None else capital
            self.accounts[agent] = Account(agent, c)
            self._log("open_account", agent, c)
        return self.accounts[agent]

    def _log(self, kind: str, agent: str | None, amount: float, **kw) -> None:
        self.journal.append({"kind": kind, "agent": agent, "amount": amount, **kw})

    # ---------- markets ----------
    def add_market(self, m: LMSRMarket) -> LMSRMarket:
        if m.market_id in self.markets:
            raise ValueError(f"duplicate market {m.market_id}")
        if len(self.open_markets()) >= self.max_markets:
            raise RuntimeError(f"market cap of {self.max_markets} reached")
        self.markets[m.market_id] = m
        self._log("open_market", None, 0.0, market=m.market_id, subsidy_bound=m.max_maker_loss)
        return m

    def open_market(
        self,
        market_id: str,
        question: str,
        outcomes: list[str] | None = None,
        b: float | None = None,
        **meta,
    ) -> LMSRMarket:
        if market_id in self.markets:
            return self.markets[market_id]
        m = LMSRMarket(market_id, question, outcomes or ["NO", "YES"], b or self.default_b, meta=meta)
        return self.add_market(m)

    def open_markets(self) -> list[LMSRMarket]:
        return [m for m in self.markets.values() if m.is_open]

    @property
    def subsidy_bound(self) -> float:
        return sum(m.max_maker_loss for m in self.markets.values())

    # ---------- trading ----------
    def buy_vector(self, agent: str, market_id: str, d: list[float], step: int = 0) -> Trade:
        acct = self.accounts[agent]
        m = self.markets[market_id]
        cost = m.quote_vector(d)
        if cost > acct.capital + 1e-9:
            raise InsufficientFunds(f"{agent} needs {cost:.2f}, has {acct.capital:.2f}")
        t = m.buy_vector(agent, d, step)
        acct.capital -= t.cost
        self._log("buy", agent, -t.cost, market=market_id, step=step)
        return t

    def buy(self, agent: str, market_id: str, outcome: int, shares: float, step: int = 0) -> Trade:
        d = [0.0] * len(self.markets[market_id].outcomes)
        d[outcome] = shares
        return self.buy_vector(agent, market_id, d, step)

    def trade_toward(
        self,
        agent: str,
        market_id: str,
        belief: list[float],
        step: int = 0,
        budget: float | None = None,
        min_move: float = 1e-4,
    ) -> Trade | None:
        """Move prices toward the agent's belief, as far as its budget allows.

        The target is the log-linear mix p^(1-a) * belief^a. With unlimited
        budget a = 1, the trade that maximises a risk-neutral agent's expected
        profit under LMSR. Otherwise a is the largest value in [0, 1] whose
        cost is within budget (default: `max_fraction_per_trade` of capital).
        Shares are buy-only: the constant that leaves prices unchanged is
        chosen so no outcome is sold.
        """
        m = self.markets[market_id]
        if not m.is_open:
            return None
        if len(belief) != len(m.outcomes):
            raise ValueError("belief has wrong length")
        acct = self.accounts[agent]
        if budget is None:
            budget = acct.capital * self.max_fraction_per_trade
        budget = min(budget, acct.capital)
        if budget <= 0:
            return None
        p = m.prices()

        def vec(alpha: float) -> list[float]:
            return m.shares_to_prices(geometric_mix(p, belief, alpha))

        alpha = 1.0
        if m.quote_vector(vec(1.0)) > budget:
            lo, hi = 0.0, 1.0
            for _ in range(50):
                mid = (lo + hi) / 2
                if m.quote_vector(vec(mid)) > budget:
                    hi = mid
                else:
                    lo = mid
            alpha = lo
        d = vec(alpha)
        if max(abs(a - b) for a, b in zip(geometric_mix(p, belief, alpha), p)) < min_move:
            return None
        return self.buy_vector(agent, market_id, d, step)

    def trade_to_belief(
        self, agent: str, market_id: str, belief: float, step: int = 0, outcome: int = 1
    ) -> Trade | None:
        """Binary convenience: belief is P(outcome)."""
        m = self.markets[market_id]
        if len(m.outcomes) != 2:
            raise ValueError("trade_to_belief is for binary markets; use trade_toward")
        belief = min(max(belief, 0.01), 0.99)
        vec = [1 - belief, belief] if outcome == 1 else [belief, 1 - belief]
        return self.trade_toward(agent, market_id, vec, step)

    # ---------- settlement ----------
    def _settle_payouts(self, m: LMSRMarket, payouts: dict[str, float]) -> None:
        for agent, amount in payouts.items():
            self.accounts[agent].capital += amount
            self._log("payout", agent, amount, market=m.market_id)
        for agent, pnl in m.pnl().items():
            self.accounts[agent].trading += pnl
        self._apply_floor()

    def resolve(self, market_id: str, outcome: int) -> dict[str, float]:
        m = self.markets[market_id]
        payouts = m.resolve(outcome)
        self._settle_payouts(m, payouts)
        return payouts

    def resolve_weighted(self, market_id: str, weights: list[float]) -> dict[str, float]:
        m = self.markets[market_id]
        payouts = m.resolve_weighted(weights)
        self._settle_payouts(m, payouts)
        return payouts

    def void(self, market_id: str) -> None:
        """Cancel an unresolvable market: refund every trader's spend, no P&L."""
        m = self.markets[market_id]
        for agent, spent in m.void().items():
            self.accounts[agent].capital += spent
            self._log("refund", agent, spent, market=market_id)

    def book(self, ledger: str, rewards: dict[str, float], note: str = "") -> None:
        if ledger not in ("information", "repair"):
            raise ValueError(ledger)
        for agent, r in rewards.items():
            acct = self.register(agent)
            setattr(acct, ledger, getattr(acct, ledger) + r)
            to_capital = ledger in self.capital_from
            if to_capital:
                acct.capital += r
            self._log(ledger, agent, r, to_capital=to_capital, note=note)
        self._apply_floor()

    def _apply_floor(self) -> None:
        for acct in self.accounts.values():
            if acct.capital < self.capital_floor:
                top = self.capital_floor - acct.capital
                acct.capital = self.capital_floor
                acct.floor_topups += top
                self._log("floor", acct.agent, top)

    # ---------- audit ----------
    def audit(self, tol: float = 1e-6) -> dict:
        """Check that every account's capital is explained by the journal."""
        bal: dict[str, float] = {}
        for e in self.journal:
            a = e["agent"]
            if a is None:
                continue
            if e["kind"] in ("information", "repair") and not e.get("to_capital"):
                continue
            bal[a] = bal.get(a, 0.0) + e["amount"]
        problems = {
            a: (bal.get(a, 0.0), acct.capital)
            for a, acct in self.accounts.items()
            if abs(bal.get(a, 0.0) - acct.capital) > tol
        }
        open_ = [m.market_id for m in self.open_markets()]
        maker = sum(m.maker_loss() for m in self.markets.values() if not m.is_open)
        return {"balanced": not problems, "problems": problems, "open_markets": open_, "maker_loss": maker}
