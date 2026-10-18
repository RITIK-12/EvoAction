"""Crusoe Managed Inference client: model discovery, budgets, caching, usage.

All calls go through `CrusoeClient.chat`, which
  1. checks the call/spend/time budget (denials are persisted as decisions),
  2. serves deterministic cache hits keyed on model + full input + scope,
  3. retries at most `MAX_RETRIES_PER_CALL` times on transport/JSON failure,
  4. persists provider-reported usage and cost for every attempt.
Provider reasoning tokens are counted but their text is never persisted.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from openai import OpenAI, RateLimitError

from evoaction.config import Settings, get_settings
from evoaction.persistence import Store, get_store, new_id

ROLE_NEMOTRON = "nemotron"
ROLE_VISION = "vision"

IMAGE_TOKEN_ESTIMATE = 1800  # conservative per-image prompt-token upper bound for budgeting


class BudgetExceeded(RuntimeError):
    pass


class ModelCallError(RuntimeError):
    raw_text: str = ""  # last reply that could not be parsed as JSON, if any


def _norm_model(mid: str) -> str:
    m = mid.lower().split("/")[-1]
    return m.removeprefix("nvidia-")


def image_part(jpeg_bytes: bytes) -> dict[str, Any]:
    b64 = base64.b64encode(jpeg_bytes).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in a model response (tolerates fences/prose)."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (t.find("{"), t.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON found in model output")
    start = min(starts)
    opener = t[start]
    closer = "}" if opener == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        c = t[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return json.loads(t[start : i + 1])
    raise ValueError("unterminated JSON in model output")


@dataclass
class ChatResult:
    text: str
    data: Any = None
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    cache_hit: bool = False
    call_id: str = ""


@dataclass
class BudgetTracker:
    settings: Settings
    store: Store
    phase: str = "optimize"  # "optimize" keeps the sealed-test vision reserve untouched; "test" may use it
    deadline_unix: float | None = None
    calls: dict[str, int] = field(default_factory=lambda: {ROLE_NEMOTRON: 0, ROLE_VISION: 0})
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def global_spend(self) -> float:
        rows = self.store.query("SELECT COALESCE(SUM(cost_usd), 0) AS s FROM model_calls WHERE cache_hit = 0")
        return float(rows[0]["s"])

    def limit_for(self, role: str) -> int:
        if role == ROLE_NEMOTRON:
            return self.settings.max_nemotron_calls
        limit = self.settings.max_vision_calls
        if self.phase != "test":
            limit -= self.settings.reserved_test_vision_calls
        return limit

    def reserve(self, run_id: str | None, role: str, model: str, projected_cost: float) -> None:
        with self._lock:
            reason = None
            if self.calls[role] >= self.limit_for(role):
                reason = f"{role} call limit reached ({self.calls[role]}/{self.limit_for(role)}, phase={self.phase})"
            elif self.global_spend() + projected_cost > self.settings.max_spend_usd:
                reason = f"projected spend would exceed MAX_SPEND_USD={self.settings.max_spend_usd}"
            elif self.deadline_unix and time.time() > self.deadline_unix:
                reason = "wall-clock stop condition reached"
            self.store.insert(
                "budget_decisions",
                {"id": new_id("BUD"), "run_id": run_id, "role": role, "model": model, "allowed": reason is None,
                 "reason": reason or "ok", "projected_cost_usd": projected_cost, "created_at": time.time()},
            )
            if reason:
                self.store.event(run_id, "budget_denied", {"role": role, "model": model, "reason": reason})
                raise BudgetExceeded(reason)
            self.calls[role] += 1


class CrusoeClient:
    def __init__(self, settings: Settings | None = None, store: Store | None = None):
        self.settings = settings or get_settings()
        if not self.settings.api_key:
            raise ModelCallError("CRUSOE_API_KEY is not set")
        self.store = store or get_store()
        self.read_cache = True  # the live demo sets False so every call really hits Crusoe
        self.client = OpenAI(
            base_url=self.settings.api_base,
            api_key=self.settings.api_key,
            timeout=self.settings.request_timeout_seconds,
            max_retries=0,
        )
        self.budget = BudgetTracker(self.settings, self.store)
        self._models: list[dict[str, Any]] | None = None
        self.run_id: str | None = None

    # -- discovery -------------------------------------------------------
    def list_models(self) -> list[dict[str, Any]]:
        if self._models is None:
            r = httpx.get(
                f"{self.settings.api_base.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {self.settings.api_key}"},
                timeout=30,
            )
            r.raise_for_status()
            self._models = r.json()["data"]
        return self._models

    def resolve_model(self, configured: str) -> str:
        ids = [m["id"] for m in self.list_models()]
        if configured in ids:
            return configured
        matches = [i for i in ids if _norm_model(i) == _norm_model(configured)]
        if len(matches) == 1:
            return matches[0]
        raise ModelCallError(f"model {configured!r} not served; available: {ids}")

    def model_info(self, model_id: str) -> dict[str, Any]:
        for m in self.list_models():
            if m["id"] == model_id:
                return m
        return {}

    def pricing(self, model_id: str) -> tuple[float, float] | None:
        p = self.model_info(model_id).get("pricing") or {}
        try:
            return float(p["prompt"]), float(p["completion"])  # USD per 1M tokens
        except (KeyError, TypeError, ValueError):
            return None

    @property
    def nemotron_model(self) -> str:
        return self.resolve_model(self.settings.reasoning_model)

    @property
    def vision_model(self) -> str:
        return self.resolve_model(self.settings.vision_model)

    # -- calls -----------------------------------------------------------
    def _estimate_cost(self, model: str, prompt_tokens: float, completion_tokens: float) -> tuple[float, bool]:
        p = self.pricing(model)
        if p is None:
            # Unknown price: assume an expensive model rather than free.
            return (prompt_tokens * 3.0 + completion_tokens * 15.0) / 1e6, True
        return (prompt_tokens * p[0] + completion_tokens * p[1]) / 1e6, False

    def chat(
        self,
        role: str,
        messages: list[dict[str, Any]],
        *,
        purpose: str,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        json_mode: bool = False,
        cache_scope: str = "",
        extra_body: dict[str, Any] | None = None,
        version_id: str | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        use_cache: bool = True,
        retries: int | None = None,
    ) -> ChatResult:
        model = model or (self.nemotron_model if role == ROLE_NEMOTRON else self.vision_model)
        params = {"max_tokens": max_tokens, "temperature": temperature, "json_mode": json_mode, "extra_body": extra_body or {}}
        key = hashlib.sha256(json.dumps([model, messages, params, cache_scope], sort_keys=True, default=str).encode()).hexdigest()
        n_images = sum(1 for m in messages if isinstance(m.get("content"), list) for p in m["content"] if p.get("type") == "image_url")

        if use_cache and self.read_cache:
            cached = self.store.cache_get(key)
            if cached is not None:
                res = ChatResult(**{**cached, "cache_hit": True, "call_id": new_id("CALL")})
                self._log_call(res, role, purpose, model, key, "ok", None, n_images, False, version_id, episode_id, task_id)
                return res

        text_chars = sum(len(m["content"]) if isinstance(m.get("content"), str) else
                         sum(len(p.get("text", "")) for p in m["content"]) for m in messages)
        projected, _ = self._estimate_cost(model, text_chars / 3.0 + n_images * IMAGE_TOKEN_ESTIMATE, max_tokens)

        last_err: Exception | None = None
        last_text = ""
        for attempt in range((self.settings.max_retries_per_call if retries is None else retries) + 1):
            self.budget.reserve(self.run_id, role, model, projected)
            t0 = time.time()
            call_id = new_id("CALL")
            try:
                kwargs: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                if extra_body:
                    kwargs["extra_body"] = extra_body
                resp = self._create_with_backoff(kwargs, role, purpose, model)
                latency = time.time() - t0
                usage = resp.usage
                pt = getattr(usage, "prompt_tokens", 0) or 0
                ct = getattr(usage, "completion_tokens", 0) or 0
                details = getattr(usage, "completion_tokens_details", None)
                rt = (getattr(details, "reasoning_tokens", 0) or 0) if details else 0
                cost, is_est = self._estimate_cost(model, pt, ct)
                text = resp.choices[0].message.content or ""
                res = ChatResult(text=text, model=model, prompt_tokens=pt, completion_tokens=ct, reasoning_tokens=rt,
                                 cost_usd=cost, latency_s=latency, call_id=call_id)
                if json_mode:
                    try:
                        res.data = extract_json(text)
                    except (ValueError, json.JSONDecodeError) as e:
                        self._log_call(res, role, purpose, model, key, "invalid_json", str(e), n_images, is_est, version_id, episode_id, task_id)
                        last_err, last_text = e, text
                        continue
                self._log_call(res, role, purpose, model, key, "ok", None, n_images, is_est, version_id, episode_id, task_id)
                self.store.cache_put(key, model, {k: v for k, v in res.__dict__.items() if k not in ("cache_hit", "call_id")})
                return res
            except BudgetExceeded:
                raise
            except Exception as e:  # transport, rate limit, server errors
                last_err = e
                res = ChatResult(text="", model=model, latency_s=time.time() - t0, call_id=call_id)
                self._log_call(res, role, purpose, model, key, "error", f"{type(e).__name__}: {e}"[:500], n_images, True, version_id, episode_id, task_id)
                if "429" in str(e) or "rate" in str(e).lower():
                    time.sleep(5 * (attempt + 1))
        err = ModelCallError(f"{purpose}: failed after retries: {last_err}")
        err.raw_text = last_text
        raise err

    def _create_with_backoff(self, kwargs: dict[str, Any], role: str, purpose: str, model: str, waits: tuple[int, ...] = (5, 10, 20, 40, 60)):
        """HTTP 429 is provider back-pressure, not a model failure: back off without spending the retry budget."""
        for i, wait in enumerate((*waits, None)):
            try:
                return self.client.chat.completions.create(**kwargs)
            except RateLimitError as e:
                self.store.event(self.run_id, "rate_limited", {"role": role, "purpose": purpose, "model": model, "wait_s": wait,
                                                               "attempt": i + 1, "error": str(e)[:200]})
                if wait is None:
                    raise
                time.sleep(wait)

    def _log_call(self, res: ChatResult, role: str, purpose: str, model: str, key: str, status: str,
                  error: str | None, n_images: int, is_est: bool, version_id, episode_id, task_id) -> None:
        self.store.insert("model_calls", {
            "id": res.call_id, "run_id": self.run_id, "role": role, "purpose": purpose, "model": model,
            "cache_key": key, "cache_hit": res.cache_hit, "status": status, "error": error,
            "prompt_tokens": 0 if res.cache_hit else res.prompt_tokens,
            "completion_tokens": 0 if res.cache_hit else res.completion_tokens,
            "reasoning_tokens": 0 if res.cache_hit else res.reasoning_tokens,
            "images": n_images, "cost_usd": 0.0 if res.cache_hit else res.cost_usd, "cost_is_estimate": is_est,
            "latency_s": res.latency_s, "version_id": version_id, "episode_id": episode_id, "task_id": task_id,
            "created_at": time.time(),
        })
