"""Live Gemma agent backed by a local model server (Ollama by default).

This is the path for running the mechanism with Gemma generating beliefs
online instead of reading cached forecasts. It was NOT used for any reported
result (the reference machine lacked disk space for the weights); it is
covered by unit tests with a fake backend. Expected setup:

    ollama pull gemma4:e4b
    python -m marketdebug live-smoke --model gemma4:e4b

The agent asks Gemma for a probability per candidate label as JSON, then
validates and normalises it. Malformed replies fall back to the market's
current prices (the agent abstains), and every reply is logged.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Protocol

from ..protocol import Observation, Reveal, TradeDistribution
from .cached import localization_market

LABELS = "ABCDEFGHIJKLMNOPQRST"


class Backend(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass
class OllamaBackend:
    model: str = "gemma4:e4b"
    host: str = "http://localhost:11434"
    temperature: float = 0.0
    timeout: float = 120.0

    def generate(self, prompt: str) -> str:
        import requests  # optional dependency: pip install marketdebug[live]

        r = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": self.temperature},
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["response"]


def build_prompt(task: dict, specialist: str, note: str, public_notes: list[tuple[str, str]], k: int) -> str:
    cands = "\n".join(
        f"{LABELS[i]}. {c.get('path', '?')}::{c.get('qualname', '?')}" for i, c in enumerate(task["candidates"][:k])
    )
    public = "\n".join(f"- [{who}] {text}" for who, text in public_notes) or "(none)"
    return (
        "You are a debugging specialist. One of the candidate functions below contains the bug.\n\n"
        f"Issue:\n{task.get('issue', '')}\n\nCandidates:\n{cands}\n\n"
        f"Your private note ({specialist}):\n{note}\n\nNotes already shared by others:\n{public}\n\n"
        f'Reply with JSON only: {{"probs": {{"A": p, ...}}}} giving a probability for every label '
        f"{LABELS[0]}-{LABELS[k-1]}, summing to 1."
    )


def parse_probs(reply: str, k: int) -> list[float] | None:
    """Extract a valid distribution over k labels, or None."""
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    probs = obj.get("probs", obj)
    if not isinstance(probs, dict):
        return None
    vals = []
    for lab in LABELS[:k]:
        v = probs.get(lab, 0.0)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        if v < 0 or v != v:
            return None
        vals.append(v)
    z = sum(vals)
    if z <= 0:
        return None
    eps = 1e-4  # keep every outcome tradeable
    return [(v / z) * (1 - eps * k) + eps for v in vals]


@dataclass
class LiveGemmaAgent:
    name: str
    specialist: str
    note: str
    backend: Backend
    per_turn_fraction: float = 0.2 / 3
    prompt_builder: Callable = build_prompt
    replies: list[dict] = field(default_factory=list)

    def act(self, obs: Observation):
        m = localization_market(obs)
        actions = []
        if m is not None:
            k = len(m.outcomes)
            public = [(who, text) for f, text, who in obs.public_facts if who != self.name]
            prompt = self.prompt_builder(obs.task, self.specialist, self.note, public, k)
            reply = self.backend.generate(prompt)
            belief = parse_probs(reply, k)
            self.replies.append({"step": obs.step, "reply": reply, "parsed": belief is not None})
            if belief is not None:
                actions.append(TradeDistribution(m.market_id, tuple(belief), budget=obs.capital * self.per_turn_fraction))
        fact = f"note:{self.specialist}"
        if all(f != fact for f, _, _ in obs.public_facts):
            actions.append(Reveal(fact, self.note))
        return actions
