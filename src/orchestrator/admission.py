"""Adaptive admission control: find the concurrency that maximises throughput.

Why not a fixed limit: generating tokens is bound by memory bandwidth (every
step reads the model's active weights). Batching more requests shares each
weight read across more tokens, so total throughput rises with concurrency
until compute saturates or the KV cache (also read every step, and growing
with each request) eats the bandwidth back. Where that peak sits depends on
the model, quantisation, context lengths and card, so it has to be measured,
per server, while it runs.

How it measures: after each change of limit, the controller first lets one
full cohort of requests (``limit`` completions) finish, so the server
settles at the new level. It then counts output tokens completed over the
next cohort and divides by the time taken. That is the server's total
throughput at this limit, measured directly. Results are kept per limit
(exponentially weighted, so they track changes in workload), which builds up
the server's speed curve as it runs. A measurement only counts if the server
was actually kept near the limit during it.

How it decides, in two phases:

1. **Probe**: double the limit each time enough samples have arrived at the
   current level, while throughput keeps up with the best seen. Doubling
   skips over local dips (on some curves 2 requests are slower in total
   than 1). Stop when throughput falls well below the best (``probe_drop``)
   or the limit reaches ``max_limit``.
2. **Tune**: run at the best measured level, and periodically try one step
   above and one step below it (about 1/8 of the limit). If a neighbour
   measures better, it becomes the new best.

Where throughput plateaus, the lowest level on the plateau wins: more
concurrency there only adds latency. Not enough ready work to fill the
server? Then nothing changes. Pass ``initial_curve`` (a previous run's ``curve``) to skip probing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class LimitSample:
    at: float
    limit: int
    throughput: float  # estimated total output tok/s at this limit
    phase: str


@dataclass
class AdaptiveLimit:
    initial: int = 4
    min_limit: int = 1
    max_limit: int = 64
    # Completions at the current level before deciding: at least this many,
    # and at least `limit` of them, so a whole cohort of requests (not just
    # the quickest to finish) is measured.
    samples_per_step: int = 6
    probe_drop: float = 0.15  # stop probing when throughput < best * (1 - this)
    smoothing: float = 0.5  # weight of a new measurement in its limit's average
    min_fill: float = 0.8  # measurement ignored if average in-flight < limit * this
    plateau: float = 0.05  # levels within this of the best count as equally good
    initial_curve: dict[int, float] | None = None  # warm start from a past run
    clock: Callable[[], float] = time.monotonic

    limit: int = field(init=False)
    phase: str = field(init=False, default="probe")
    history: list[LimitSample] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if not (1 <= self.min_limit <= self.max_limit):
            raise ValueError("need 1 <= min_limit <= max_limit")
        self.limit = self._clamp(self.initial)
        # limit -> measured total output tok/s
        self.curve: dict[int, float] = dict(self.initial_curve or {})
        self._in_flight = 0
        self._area = 0.0  # integral of in-flight over time
        self._last = self.clock()
        self._best_probe = 0.0
        self._center = self.limit
        self._explore = [1, 0, -1, 0]  # tune cycle: up, center, down, center
        self._explore_i = 0
        if self.curve:
            # Warm start: skip probing and begin at the best known level.
            self.phase = "tune"
            self._center = self.limit = self._clamp(self.best_level())
        self._begin_settle()

    def _clamp(self, n: int) -> int:
        return max(self.min_limit, min(self.max_limit, n))

    def _cohort(self) -> int:
        return max(self.samples_per_step, self.limit)

    def _begin_settle(self) -> None:
        self._settle_left = self._cohort()
        self._win_start: float | None = None

    def _advance(self) -> float:
        now = self.clock()
        self._area += self._in_flight * (now - self._last)
        self._last = now
        return now

    # -- hooks called by the pool -----------------------------------------

    def on_start(self) -> None:
        self._advance()
        self._in_flight += 1

    def on_complete(self, output_tokens: int) -> None:
        now = self._advance()
        self._in_flight -= 1
        if self._settle_left > 0:
            self._settle_left -= 1
            if self._settle_left == 0:
                self._win_start, self._win_area0 = now, self._area
                self._win_tokens, self._win_done = 0, 0
            return
        self._win_tokens += output_tokens
        self._win_done += 1
        if self._win_done < self._cohort():
            return
        elapsed = now - self._win_start
        avg_in_flight = (self._area - self._win_area0) / elapsed if elapsed > 0 else 0.0
        if elapsed > 0 and avg_in_flight >= self.min_fill * self.limit:
            tps = self._win_tokens / elapsed
            old = self.curve.get(self.limit)
            self.curve[self.limit] = tps if old is None else old + self.smoothing * (tps - old)
            before = self.limit
            self._decide(now, tps)
            if self.limit != before:
                self._begin_settle()
                return
        # Same limit (or not enough demand to judge): measure the next cohort.
        self._win_start, self._win_area0 = now, self._area
        self._win_tokens, self._win_done = 0, 0

    # -- control ----------------------------------------------------------

    def best_level(self) -> int | None:
        """Lowest concurrency whose throughput is within ``plateau`` of the best.

        Past the point where throughput stops rising (for example when the KV
        cache is full and extra requests just queue inside the server), more
        concurrency only adds latency, so prefer the low end of the plateau.
        """
        if not self.curve:
            return None
        top = max(self.curve.values())
        return min(lvl for lvl, v in self.curve.items() if v >= top * (1 - self.plateau))

    def _decide(self, now: float, tps: float) -> None:
        self.history.append(LimitSample(now, self.limit, tps, self.phase))

        if self.phase == "probe":
            self._best_probe = max(self._best_probe, tps)
            if tps >= self._best_probe * (1 - self.probe_drop) and self.limit < self.max_limit:
                self.limit = self._clamp(self.limit * 2)
                return
            self.phase = "tune"
            self._center = self._clamp(self.best_level() or self.limit)
            self.limit = self._center
            self._explore_i = 0
            return

        # Tune: move the center to the best measured level near it, then try
        # the next point in the exploration cycle.
        best = self.best_level()
        if best is not None:
            self._center = self._clamp(best)
        step = max(1, self._center // 8)
        offset = self._explore[self._explore_i % len(self._explore)]
        self._explore_i += 1
        self.limit = self._clamp(self._center + offset * step)

    @property
    def best(self) -> tuple[int, float] | None:
        """(concurrency level, estimated total output tok/s) of the chosen best level."""
        lvl = self.best_level()
        return None if lvl is None else (lvl, self.curve[lvl])
