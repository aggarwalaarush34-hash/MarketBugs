"""Shared evidence board.

Every report is timestamped and attributed. A report only *counts* toward
information rewards if its fact was not already public when it was filed;
otherwise a copycat would split an original finder's Shapley credit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Report:
    report_id: str
    agent: str
    step: int
    fact: str          # canonical key for what was observed (dedupe key)
    text: str = ""
    counts: bool = True


@dataclass
class EvidenceBoard:
    # Maps free text to a canonical fact key. In a simulation the agent passes
    # the key directly; with LLM agents plug in a judge/embedding matcher here.
    canonicalize: Callable[[str, str], str] = lambda fact, text: fact
    reports: list[Report] = field(default_factory=list)
    _first: dict[str, Report] = field(default_factory=dict)

    def post(self, agent: str, fact: str, text: str = "", step: int = 0) -> Report:
        key = self.canonicalize(fact, text)
        original = key not in self._first
        r = Report(f"r{len(self.reports)}", agent, step, key, text, counts=original)
        self.reports.append(r)
        if original:
            self._first[key] = r
        return r

    def public_facts(self) -> set[str]:
        return set(self._first)

    def counted(self) -> list[Report]:
        return [r for r in self.reports if r.counts]

    def by_agent(self) -> dict[str, list[Report]]:
        out: dict[str, list[Report]] = {}
        for r in self.counted():
            out.setdefault(r.agent, []).append(r)
        return out
