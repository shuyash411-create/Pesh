"""Provider adapters: run(model, task) -> RunResult, over HTTPS APIs only (nothing runs locally).

Keys come from the environment and are only ever placed in request headers. They are never
logged, never stored, hidden from ``repr`` and scrubbed from error messages. A provider whose
key is missing reports ``available() == False`` and is skipped by the runner.

Cost is what the API actually reports (token usage) times the per-token list price in
``DEFAULT_MODELS`` (override with ``--prices``). Provider APIs return tokens, not dollars.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import httpx

from ..config import DGPParams, PricingParams


class ProviderError(RuntimeError):
    """A failed API call (message is already scrubbed of credentials)."""


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    provider: str           # anthropic | openai | together | fireworks | sim
    model_id: str
    price_in: float         # USD per input token
    price_out: float        # USD per output token

    def cost(self, input_tokens: float, output_tokens: float) -> float:
        return self.price_in * input_tokens + self.price_out * output_tokens


@dataclass
class RunResult:
    input_tokens: int
    output_tokens: int
    cost: float
    wall_ms: float
    steps: int
    success: bool           # did the task complete; NOT a quality score

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Task:
    task_id: str
    text: str
    features: dict


def _mtok(i: float, o: float) -> tuple[float, float]:
    return i / 1e6, o / 1e6


# Default aliases. List prices change; check them and override with --prices before trusting $.
DEFAULT_MODELS: dict[str, ModelSpec] = {
    "claude": ModelSpec("claude", "anthropic", "claude-haiku-4-5-20251001", *_mtok(1.00, 5.00)),
    "gpt": ModelSpec("gpt", "openai", "gpt-4o-mini", *_mtok(0.15, 0.60)),
    "together-open": ModelSpec("together-open", "together", "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                               *_mtok(0.88, 0.88)),
    "fireworks-open": ModelSpec("fireworks-open", "fireworks",
                                "accounts/fireworks/models/llama-v3p3-70b-instruct", *_mtok(0.90, 0.90)),
}

KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
           "together": "TOGETHER_API_KEY", "fireworks": "FIREWORKS_API_KEY"}


def resolve_models(names: list[str], prices_file: str | None = None) -> list[ModelSpec]:
    """Turn ``claude,gpt`` or ``provider:model-id`` entries into ModelSpecs.

    ``prices_file`` is JSON ``{alias_or_model_id: {"in_per_mtok": x, "out_per_mtok": y}}``; it
    overrides the defaults and is required for ``provider:model-id`` entries.
    """
    over = json.loads(Path(prices_file).read_text()) if prices_file else {}
    out = []
    for name in names:
        if name in DEFAULT_MODELS:
            spec = DEFAULT_MODELS[name]
            model_id = spec.model_id
        elif ":" in name and name.split(":", 1)[0] in KEY_ENV:
            prov, model_id = name.split(":", 1)
            spec = ModelSpec(name, prov, model_id, float("nan"), float("nan"))
        else:
            raise ValueError(f"unknown model {name!r}; use one of {sorted(DEFAULT_MODELS)} or provider:model-id "
                             f"(providers: {sorted(KEY_ENV)})")
        p = over.get(name) or over.get(model_id)
        if p:
            spec = ModelSpec(spec.alias, spec.provider, spec.model_id, p["in_per_mtok"] / 1e6,
                             p["out_per_mtok"] / 1e6)
        if spec.price_in != spec.price_in:
            raise ValueError(f"no price known for {name!r}; add it to --prices")
        out.append(spec)
    return out


# ----------------------------------------------------------------------------- adapters

class Provider:
    name = "base"

    def available(self) -> bool:
        raise NotImplementedError

    def run(self, model: ModelSpec, task: Task, run_idx: int = 0, kind: str = "cross") -> RunResult:
        raise NotImplementedError


class _HTTPProvider(Provider):
    url = ""

    def __init__(self, env: dict | None = None, client: httpx.Client | None = None, max_tokens: int = 512,
                 timeout: float = 120.0, retries: int = 2):
        self._key = (env if env is not None else os.environ).get(KEY_ENV[self.name], "").strip()
        self._client = client or httpx.Client(timeout=timeout)
        self.max_tokens = max_tokens
        self.retries = retries

    def __repr__(self) -> str:
        return f"{type(self).__name__}(key={'set' if self._key else 'missing'})"

    def available(self) -> bool:
        return bool(self._key)

    def _scrub(self, s: str) -> str:
        return s.replace(self._key, "***") if self._key else s

    def _headers(self) -> dict:
        raise NotImplementedError

    def _body(self, model: ModelSpec, task: Task) -> dict:
        raise NotImplementedError

    def _parse(self, data: dict) -> tuple[int, int, bool]:
        """-> (input_tokens, output_tokens, produced_output)"""
        raise NotImplementedError

    def run(self, model: ModelSpec, task: Task, run_idx: int = 0, kind: str = "cross") -> RunResult:
        if not self.available():
            raise ProviderError(f"{KEY_ENV[self.name]} is not set")
        t0 = time.perf_counter()
        delay = 2.0
        for attempt in range(self.retries + 1):
            try:
                r = self._client.post(self.url, headers=self._headers(), json=self._body(model, task))
            except httpx.HTTPError as e:
                if attempt < self.retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise ProviderError(self._scrub(f"{type(e).__name__}: {e}")) from None
            if r.status_code in (429, 500, 502, 503, 529) and attempt < self.retries:
                time.sleep(delay)
                delay *= 2
                continue
            break
        wall = (time.perf_counter() - t0) * 1000.0
        if r.status_code != 200:
            raise ProviderError(self._scrub(f"HTTP {r.status_code}: {r.text[:300]}"))
        try:
            tin, tout, produced = self._parse(r.json())
        except (KeyError, TypeError, ValueError, IndexError) as e:
            raise ProviderError(f"unexpected response shape ({type(e).__name__})") from None
        return RunResult(tin, tout, model.cost(tin, tout), wall, 1, produced)


class AnthropicProvider(_HTTPProvider):
    name = "anthropic"
    url = "https://api.anthropic.com/v1/messages"

    def _headers(self):
        return {"x-api-key": self._key, "anthropic-version": "2023-06-01", "content-type": "application/json"}

    def _body(self, model, task):
        return {"model": model.model_id, "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": task.text}]}

    def _parse(self, data):
        u = data["usage"]
        # cache reads/writes are billed differently; count them as plain input (conservative)
        tin = int(u["input_tokens"]) + int(u.get("cache_read_input_tokens") or 0) \
            + int(u.get("cache_creation_input_tokens") or 0)
        produced = any(b.get("type") == "text" and b.get("text") for b in data.get("content", []))
        return tin, int(u["output_tokens"]), produced


class _OpenAICompat(_HTTPProvider):
    def _headers(self):
        return {"authorization": f"Bearer {self._key}", "content-type": "application/json"}

    def _body(self, model, task):
        return {"model": model.model_id, "max_completion_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": task.text}]}

    def _parse(self, data):
        u = data["usage"]
        msg = data["choices"][0]["message"]
        return int(u["prompt_tokens"]), int(u["completion_tokens"]), bool(msg.get("content"))


class OpenAIProvider(_OpenAICompat):
    name = "openai"
    url = "https://api.openai.com/v1/chat/completions"


class TogetherProvider(_OpenAICompat):
    name = "together"
    url = "https://api.together.xyz/v1/chat/completions"

    def _body(self, model, task):
        b = super()._body(model, task)
        b["max_tokens"] = b.pop("max_completion_tokens")
        return b


class FireworksProvider(_OpenAICompat):
    name = "fireworks"
    url = "https://api.fireworks.ai/inference/v1/chat/completions"

    def _body(self, model, task):
        b = super()._body(model, task)
        b["max_tokens"] = b.pop("max_completion_tokens")
        return b


class SimProvider(Provider):
    """Offline fallback: the existing calibrated simulator (``pesh.sim.dgp``) plays the API.

    Each (task, model, run) gets its own seed, so repeated runs of one task vary run-to-run
    while sharing the task's latent difficulty (which is what makes the ICC meaningful).
    Tokens are billed at the model's own input/output price (cache multipliers as in the simulator).
    """

    name = "sim"

    def __init__(self, dgp: DGPParams | None = None):
        self.dgp = dgp or DGPParams()

    def available(self) -> bool:
        return True

    def run(self, model: ModelSpec, task: Task, run_idx: int = 0, kind: str = "cross") -> RunResult:
        import zlib

        import numpy as np
        import pandas as pd

        from ..sim.dgp import execute, plan_runs

        p, base = self.dgp, PricingParams()
        f = task.features
        x = np.array([float(f.get("x_repo", 0.0)), float(f.get("x_complex", 0.0))])
        # task-level latent difficulty: fixed per task, independent of model / run
        u = np.random.default_rng(zlib.crc32(task.task_id.encode())).normal(0.0, p.sigma_u)
        d = float(x @ np.asarray(p.theta) + u)
        tasks = pd.DataFrame({"x_repo": [x[0]], "x_complex": [x[1]], "difficulty": [d]})
        seed = zlib.crc32(f"{task.task_id}|{model.alias}|{kind}|{run_idx}".encode())
        t0 = time.perf_counter()
        res = execute(plan_runs(tasks, 1, p, seed=seed), params=p, pricing=replace(base, p_in=model.price_in, p_out=model.price_out)).runs.iloc[0]
        return RunResult(int(res["input_tokens"]), int(res["output_tokens"]), float(res["cost"]),
                         (time.perf_counter() - t0) * 1000.0, int(res["steps"]), bool(res["success"]))


ADAPTERS = {"anthropic": AnthropicProvider, "openai": OpenAIProvider,
            "together": TogetherProvider, "fireworks": FireworksProvider}


def build_providers(env: dict | None = None, **kw) -> dict[str, Provider]:
    """Adapters for every provider whose key is set in ``env`` (default: os.environ)."""
    out = {}
    for name, cls in ADAPTERS.items():
        p = cls(env=env, **kw)
        if p.available():
            out[name] = p
    return out
