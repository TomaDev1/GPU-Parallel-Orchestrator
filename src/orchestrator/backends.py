"""Inference backends.

The scheduler only needs ``await backend.complete(request)``. Concurrency is
the point: the scheduler keeps several requests in flight at once and a
batching server (Strata, vLLM, SGLang, llama.cpp server) merges them into
shared GPU passes.
"""

from __future__ import annotations

import asyncio
import bisect
import time
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Request:
    prompt: str
    system: str | None = None
    max_tokens: int = 512
    task_id: str | None = None


@dataclass
class Completion:
    text: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    # Time from request start until the first output token, in seconds.
    ttft: float | None = None


class Backend(Protocol):
    async def complete(self, request: Request) -> Completion: ...


# ---------------------------------------------------------------------------
# Real servers
# ---------------------------------------------------------------------------


class OpenAICompatibleBackend:
    """Calls ``POST {base_url}/chat/completions``.

    Works with any OpenAI-compatible server: vLLM, SGLang, llama.cpp server,
    LM Studio, or Strata if/when it exposes that API.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 600.0,
        temperature: float = 0.7,
        max_connections: int = 64,
        transport=None,  # httpx transport override, mainly for tests
    ) -> None:
        try:
            import httpx
        except ImportError as e:  # pragma: no cover - depends on environment
            raise ImportError("OpenAICompatibleBackend needs httpx: pip install httpx") from e
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            limits=httpx.Limits(max_connections=max_connections),
            transport=transport,
        )
        self.model = model
        self.temperature = temperature

    async def complete(self, request: Request) -> Completion:
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        resp = await self._client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": self.temperature,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage") or {}
        return Completion(
            text=data["choices"][0]["message"]["content"] or "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Simulated batching server
# ---------------------------------------------------------------------------

# Per-request decode speed (tokens/s) by number of concurrent requests. These
# defaults come from a published Strata 4-GPU reference run that has NOT been
# verified for this project; replace them with real measurements once the
# workstation is built.
DEFAULT_SPEED_CURVE: dict[int, float] = {1: 123.0, 2: 57.0, 4: 51.0, 8: 45.0}


@dataclass
class _Active:
    request: Request
    prefill_left: float  # prompt tokens still to process
    decode_left: float  # output tokens still to generate
    target_tokens: int
    started: float
    first_token_at: float | None = None
    done: asyncio.Future = field(default=None)  # type: ignore[assignment]


class SimulatedBackend:
    """A fake continuous-batching server, for developing without GPUs.

    All in-flight requests share the "GPU". Each tick, every request in decode
    advances by ``speed(n)`` tokens/s, where ``n`` is how many requests are
    decoding at once. Prompts are prefilled at ``prefill_tps`` shared across
    requests in prefill. New requests join the batch immediately (continuous
    batching).

    ``time_scale`` speeds up the simulation: 100 means one simulated second
    takes 10 ms of real time. Reported times in completions are simulated
    seconds.

    Above ``max_batch`` concurrent requests, extra requests wait in a queue,
    like a server with a fixed number of batch slots.
    """

    def __init__(
        self,
        speed_curve: dict[int, float] | None = None,
        prefill_tps: float = 2000.0,
        max_batch: int = 16,
        time_scale: float = 1.0,
        tick: float = 0.05,
        chars_per_token: float = 4.0,
    ) -> None:
        curve = dict(sorted((speed_curve or DEFAULT_SPEED_CURVE).items()))
        if not curve or min(curve) < 1:
            raise ValueError("speed_curve needs keys >= 1")
        self._curve_n = list(curve)
        self._curve_v = list(curve.values())
        self.prefill_tps = prefill_tps
        self.max_batch = max_batch
        self.time_scale = time_scale
        self.tick = tick
        self.chars_per_token = chars_per_token

        self._active: list[_Active] = []
        self._slots = asyncio.Semaphore(max_batch) if max_batch else None
        self._loop_task: asyncio.Task | None = None
        self._wake = asyncio.Event()
        self._sim_time = 0.0
        self.peak_concurrency = 0
        self.total_output_tokens = 0

    # Simulated clock -------------------------------------------------------

    def now(self) -> float:
        """Current simulated time in seconds."""
        return self._sim_time

    def speed(self, n: int) -> float:
        """Per-request decode tokens/s with n requests decoding together.

        Linear interpolation between curve points; beyond the last point,
        aggregate throughput is held flat (per-request speed falls as 1/n).
        """
        if n <= 0:
            return 0.0
        ns, vs = self._curve_n, self._curve_v
        if n <= ns[0]:
            return vs[0]
        if n >= ns[-1]:
            return vs[-1] * ns[-1] / n
        i = bisect.bisect_left(ns, n)
        n0, n1, v0, v1 = ns[i - 1], ns[i], vs[i - 1], vs[i]
        return v0 + (v1 - v0) * (n - n0) / (n1 - n0)

    def aggregate_speed(self, n: int) -> float:
        return self.speed(n) * n

    def _tokens(self, text: str) -> int:
        return max(1, int(len(text) / self.chars_per_token))

    # Request handling -----------------------------------------------------

    async def complete(self, request: Request) -> Completion:
        if self._slots:
            await self._slots.acquire()
        try:
            loop = asyncio.get_running_loop()
            prompt_tokens = self._tokens((request.system or "") + request.prompt)
            # The simulator always generates max_tokens; set it to the expected length.
            target = request.max_tokens
            act = _Active(
                request=request,
                prefill_left=prompt_tokens,
                decode_left=target,
                target_tokens=target,
                started=self._sim_time,
                done=loop.create_future(),
            )
            self._active.append(act)
            self.peak_concurrency = max(self.peak_concurrency, len(self._active))
            self._ensure_loop()
            self._wake.set()
            await act.done
            self.total_output_tokens += target
            text = f"[simulated output for {request.task_id or 'request'}: {target} tokens]"
            return Completion(
                text=text,
                prompt_tokens=prompt_tokens,
                output_tokens=target,
                ttft=(act.first_token_at or self._sim_time) - act.started,
            )
        finally:
            if self._slots:
                self._slots.release()

    def _ensure_loop(self) -> None:
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> None:
        while True:
            if not self._active:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    return  # idle; the loop restarts on the next request
                continue
            await asyncio.sleep(self.tick / self.time_scale)
            self._step(self.tick)

    def _step(self, dt: float) -> None:
        self._sim_time += dt
        prefilling = [a for a in self._active if a.prefill_left > 0]
        decoding = [a for a in self._active if a.prefill_left <= 0]
        if prefilling:
            share = self.prefill_tps * dt / len(prefilling)
            for a in prefilling:
                a.prefill_left -= share
        if decoding:
            per = self.speed(len(decoding)) * dt
            for a in decoding:
                if a.first_token_at is None:
                    a.first_token_at = self._sim_time
                a.decode_left -= per
        finished = [a for a in self._active if a.prefill_left <= 0 and a.decode_left <= 0]
        for a in finished:
            self._active.remove(a)
            if not a.done.done():
                a.done.set_result(None)
