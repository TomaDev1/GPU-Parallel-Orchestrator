"""Inference backends.

The scheduler only needs ``await backend.complete(request)``. Concurrency is
the point: the scheduler keeps several requests in flight at once and a
batching server (Strata, vLLM, SGLang, llama.cpp server) merges them into
shared GPU passes.
"""

from __future__ import annotations

import asyncio
import bisect
import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Protocol


def prefix_key_for(system: str | None) -> str | None:
    """Default cache key for a request's reusable prefix: a hash of its system prompt.

    Servers with prefix caching (vLLM, SGLang, llama.cpp) reuse the KV cache
    for a prompt's leading tokens. Agents that share a system prompt (and any
    shared context placed in it) share that prefix.
    """
    if not system:
        return None
    return hashlib.sha1(system.encode()).hexdigest()[:16]


@dataclass
class Request:
    prompt: str
    system: str | None = None
    max_tokens: int = 512
    task_id: str | None = None
    # Requests with the same key share a cacheable prefix. Defaults to a hash
    # of the system prompt.
    prefix_key: str | None = None

    def __post_init__(self) -> None:
        if self.prefix_key is None:
            self.prefix_key = prefix_key_for(self.system)


@dataclass
class Completion:
    text: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    # Time from request start until the first output token, in seconds.
    ttft: float | None = None
    # Which backend served the request (set by BackendPool).
    endpoint: str | None = None
    # Prompt tokens served from the prefix cache (simulator only).
    cached_tokens: int = 0


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


class SimClock:
    """Shared simulated clock that steps one or more SimulatedBackends together.

    Give every simulated server in a pool the same clock so their timelines
    line up, and pass ``clock.now`` to the Scheduler.

    ``time_scale`` speeds up the simulation: 100 means one simulated second
    takes 10 ms of real time. Each tick advances simulated time by ``tick``
    seconds, regardless of real-time jitter, so results are deterministic.
    """

    def __init__(self, time_scale: float = 1.0, tick: float = 0.05) -> None:
        self.time_scale = time_scale
        self.tick = tick
        self._t = 0.0
        self._backends: list["SimulatedBackend"] = []
        self._loop_task: asyncio.Task | None = None
        self._wake: asyncio.Event | None = None

    def now(self) -> float:
        return self._t

    def register(self, backend: "SimulatedBackend") -> None:
        self._backends.append(backend)

    def poke(self) -> None:
        """Called when a backend gets work; starts the tick loop if idle."""
        if self._wake is None:
            self._wake = asyncio.Event()
        self._wake.set()
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> None:
        assert self._wake is not None
        while True:
            if not any(b._active for b in self._backends):
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    return  # idle; restarted by the next poke()
                continue
            await asyncio.sleep(self.tick / self.time_scale)
            self._t += self.tick
            for b in self._backends:
                b._step(self.tick)


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

    Time is simulated (see SimClock). Pass ``clock=`` to share one clock
    between several simulated servers; otherwise each gets its own, built
    from ``time_scale`` and ``tick``.

    Above ``max_batch`` concurrent requests, extra requests wait in a queue,
    like a server with a fixed number of batch slots.

    Prefix caching: once a request with a given ``prefix_key`` has been
    prefilled, later requests with the same key skip prefilling the system
    prompt (an LRU of ``prefix_cache_size`` prefixes). Set the size to 0 to
    disable.

    KV cache: with ``kv_capacity_tokens`` set, each running request reserves
    room for its prompt plus its full output. A request that doesn't fit
    waits inside the server until room frees up, as vLLM does. This is what
    caps useful concurrency on cards with little VRAM left after the weights.
    """

    def __init__(
        self,
        speed_curve: dict[int, float] | None = None,
        prefill_tps: float = 2000.0,
        max_batch: int = 16,
        time_scale: float = 1.0,
        tick: float = 0.05,
        chars_per_token: float = 4.0,
        prefix_cache_size: int = 32,
        clock: SimClock | None = None,
        kv_capacity_tokens: int | None = None,
    ) -> None:
        curve = dict(sorted((speed_curve or DEFAULT_SPEED_CURVE).items()))
        if not curve or min(curve) < 1:
            raise ValueError("speed_curve needs keys >= 1")
        self._curve_n = list(curve)
        self._curve_v = list(curve.values())
        self.prefill_tps = prefill_tps
        self.max_batch = max_batch
        self.clock = clock or SimClock(time_scale=time_scale, tick=tick)
        self.clock.register(self)
        self.chars_per_token = chars_per_token
        self.prefix_cache_size = prefix_cache_size
        self._prefix_cache: OrderedDict[str, None] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.kv_capacity_tokens = kv_capacity_tokens
        self._kv_used = 0
        self._kv_cond: asyncio.Condition | None = None

        self._active: list[_Active] = []
        self._slots = asyncio.Semaphore(max_batch) if max_batch else None
        self.peak_concurrency = 0
        self.total_output_tokens = 0

    # Simulated clock -------------------------------------------------------

    def now(self) -> float:
        """Current simulated time in seconds."""
        return self.clock.now()

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
            system_tokens = self._tokens(request.system) if request.system else 0
            prompt_tokens = system_tokens + self._tokens(request.prompt)
            cached = system_tokens if self._cache_lookup(request.prefix_key) else 0
            # The simulator always generates max_tokens; set it to the expected length.
            target = request.max_tokens
            kv_need = prompt_tokens + target
            await self._kv_reserve(kv_need)
            act = _Active(
                request=request,
                prefill_left=prompt_tokens - cached,
                decode_left=target,
                target_tokens=target,
                started=self.now(),
                done=loop.create_future(),
            )
            self._active.append(act)
            self.peak_concurrency = max(self.peak_concurrency, len(self._active))
            self.clock.poke()
            try:
                await act.done
            finally:
                await self._kv_release(kv_need)
            self.total_output_tokens += target
            text = f"[simulated output for {request.task_id or 'request'}: {target} tokens]"
            return Completion(
                text=text,
                prompt_tokens=prompt_tokens,
                output_tokens=target,
                ttft=(act.first_token_at or self.now()) - act.started,
                cached_tokens=cached,
            )
        finally:
            if self._slots:
                self._slots.release()

    async def _kv_reserve(self, tokens: int) -> None:
        if self.kv_capacity_tokens is None:
            return
        if self._kv_cond is None:
            self._kv_cond = asyncio.Condition()
        async with self._kv_cond:
            # Always admit into an empty server, even if the request is huge.
            while self._kv_used > 0 and self._kv_used + tokens > self.kv_capacity_tokens:
                await self._kv_cond.wait()
            self._kv_used += tokens

    async def _kv_release(self, tokens: int) -> None:
        if self.kv_capacity_tokens is None or self._kv_cond is None:
            return
        async with self._kv_cond:
            self._kv_used -= tokens
            self._kv_cond.notify_all()

    def _cache_lookup(self, key: str | None) -> bool:
        """Return True on a hit; on a miss, insert the key (evicting LRU)."""
        if key is None or self.prefix_cache_size <= 0:
            return False
        if key in self._prefix_cache:
            self._prefix_cache.move_to_end(key)
            self.cache_hits += 1
            return True
        self.cache_misses += 1
        self._prefix_cache[key] = None
        while len(self._prefix_cache) > self.prefix_cache_size:
            self._prefix_cache.popitem(last=False)
        return False

    def _step(self, dt: float) -> None:
        """Advance this server by dt simulated seconds (called by SimClock)."""
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
                    a.first_token_at = self.now()
                a.decode_left -= per
        finished = [a for a in self._active if a.prefill_left <= 0 and a.decode_left <= 0]
        for a in finished:
            self._active.remove(a)
            if not a.done.done():
                a.done.set_result(None)
