"""BackendPool: spread requests over several inference servers.

The intended layout is one model copy per GPU, each behind its own server.
The pool acts like a single Backend to the scheduler and decides, per request,
which server gets it:

* Each server has its own concurrency limit (its batch slots).
* ``prefix`` routing (default): a request goes to a server that already has
  its prefix cached, if one has a free slot and is not more than
  ``max_imbalance`` requests busier than the least-loaded server. Otherwise
  it goes to the least-loaded free server, which then caches that prefix too.
  This keeps agents that share a long system prompt on the same card, so the
  prompt is processed once instead of once per card, without letting one
  card pile up work while others sit idle.
* ``least_loaded`` and ``round_robin`` are there for comparison.

Routing never waits for a "preferred" server while another is idle: it is
work-conserving, like the scheduler.
"""

from __future__ import annotations

import asyncio
import itertools
from collections import OrderedDict
from dataclasses import dataclass, field

from .backends import Backend, Completion, Request

STRATEGIES = ("prefix", "least_loaded", "round_robin")


@dataclass
class Endpoint:
    name: str
    backend: Backend
    max_concurrency: int = 16
    in_flight: int = 0
    served: int = 0
    # Prefix keys this endpoint has (probably) cached, most recent last.
    prefixes: OrderedDict = field(default_factory=OrderedDict)

    @property
    def free(self) -> bool:
        return self.in_flight < self.max_concurrency

    @property
    def load(self) -> float:
        return self.in_flight / self.max_concurrency


class BackendPool:
    def __init__(
        self,
        endpoints: list[Endpoint],
        strategy: str = "prefix",
        prefixes_per_endpoint: int = 64,
        max_imbalance: int = 4,
    ) -> None:
        if not endpoints:
            raise ValueError("BackendPool needs at least one endpoint")
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        names = [e.name for e in endpoints]
        if len(set(names)) != len(names):
            raise ValueError("endpoint names must be unique")
        self.endpoints = endpoints
        self.strategy = strategy
        self.prefixes_per_endpoint = prefixes_per_endpoint
        self.max_imbalance = max_imbalance
        self._rr = itertools.cycle(range(len(endpoints)))
        self._cond: asyncio.Condition | None = None
        self.affinity_hits = 0
        self.affinity_misses = 0

    @property
    def capacity(self) -> int:
        """Total concurrent requests the pool can run; a good scheduler limit."""
        return sum(e.max_concurrency for e in self.endpoints)

    def _condition(self) -> asyncio.Condition:
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    def choose(self, request: Request) -> Endpoint | None:
        """Pick an endpoint with a free slot, or None if all are full."""
        free = [e for e in self.endpoints if e.free]
        if not free:
            return None
        if self.strategy == "round_robin":
            for _ in range(len(self.endpoints)):
                e = self.endpoints[next(self._rr)]
                if e.free:
                    return e
        least = min(free, key=lambda e: (e.load, e.served))
        if self.strategy == "prefix" and request.prefix_key is not None:
            warm = [e for e in free if request.prefix_key in e.prefixes]
            if warm:
                best = min(warm, key=lambda e: e.load)
                if best.in_flight - least.in_flight <= self.max_imbalance:
                    self.affinity_hits += 1
                    return best
            self.affinity_misses += 1
        return least

    def _remember(self, ep: Endpoint, key: str | None) -> None:
        if key is None:
            return
        ep.prefixes[key] = None
        ep.prefixes.move_to_end(key)
        while len(ep.prefixes) > self.prefixes_per_endpoint:
            ep.prefixes.popitem(last=False)

    async def complete(self, request: Request) -> Completion:
        cond = self._condition()
        async with cond:
            while (ep := self.choose(request)) is None:
                await cond.wait()
            ep.in_flight += 1
            self._remember(ep, request.prefix_key)
        try:
            completion = await ep.backend.complete(request)
            completion.endpoint = ep.name
            ep.served += 1
            return completion
        finally:
            async with cond:
                ep.in_flight -= 1
                cond.notify_all()

    async def aclose(self) -> None:
        for e in self.endpoints:
            close = getattr(e.backend, "aclose", None)
            if close is not None:
                await close()
