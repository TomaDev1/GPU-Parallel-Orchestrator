"""Adaptive concurrency vs fixed limits on 4 simulated cards.

    python examples/demo_adaptive.py

Two scenarios, both 4 cards with one model copy each:

1. ``27B on 3080 20GB``: the QWEN27B_RTX3080_20GB preset (estimates, see
   presets.py). Throughput rises with concurrency until the small KV cache
   fills; past that, extra requests just queue inside the server.
2. ``peaked curve``: a HYPOTHETICAL card whose total throughput peaks around
   40 concurrent and then falls. It checks that the controller finds a peak
   it isn't told about.

None of this predicts real speed. Replace the presets with measurements once
the hardware is running.
"""

from __future__ import annotations

import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchestrator import (  # noqa: E402
    QWEN27B_RTX3080_20GB,
    AdaptiveLimit,
    BackendPool,
    Endpoint,
    Scheduler,
    SimClock,
    SimulatedBackend,
    Task,
    TaskGraph,
)

PEAKED = {"speed_curve": {1: 46.0, 32: 22.0, 64: 8.0}, "max_batch": 0}
TASKS = 800


def build_graph() -> TaskGraph:
    """Agent-style requests: 250-2,500 token prompts, 100-500 token answers."""
    rng = random.Random(0)
    return TaskGraph([
        Task(f"t{i}", "context " * rng.randint(125, 1250), max_tokens=rng.randint(100, 500))
        for i in range(TASKS)
    ])


async def run(label: str, make_endpoint, sim_kwargs: dict) -> None:
    clock = SimClock(time_scale=2000, tick=0.1)
    sims = [SimulatedBackend(clock=clock, **sim_kwargs) for _ in range(4)]
    pool = BackendPool([make_endpoint(f"gpu{i}", s, clock) for i, s in enumerate(sims)], clock=clock.now)
    report = await Scheduler(pool, max_concurrency=pool.capacity, clock=clock.now).run(build_graph())
    extra = ""
    ctrl = pool.endpoints[0].controller
    if ctrl:
        extra = f" | gpu0 settled near {ctrl.best[0]} concurrent"
    print(f"  {label:<14} makespan {report.makespan:7.1f}s | {report.throughput:7.1f} tok/s{extra}")


async def scenario(title: str, sim_kwargs: dict) -> None:
    print(f"{title}: {TASKS} agent requests on 4 cards")
    for fixed in (4, 8, 16, 32, 64):
        await run(f"fixed {fixed}", lambda n, s, c, f=fixed: Endpoint(n, s, max_concurrency=f), sim_kwargs)
    # Cold: the controllers start knowing nothing about the cards.
    controllers: dict[str, AdaptiveLimit] = {}

    def adaptive(name, sim, clock):
        # Warm start from what this card's controller learned last run, if any.
        learned = controllers[name].curve if name in controllers else None
        ctrl = controllers[name] = AdaptiveLimit(
            initial=4, max_limit=64, initial_curve=learned, clock=clock.now
        )
        return Endpoint(name, sim, controller=ctrl)

    await run("adaptive cold", adaptive, sim_kwargs)
    # Warm: same controllers on the next job, keeping what they learned.
    await run("adaptive warm", adaptive, sim_kwargs)
    print()


async def main() -> None:
    await scenario("27B on 3080 20GB (estimate)", QWEN27B_RTX3080_20GB)
    await scenario("Peaked curve (hypothetical)", PEAKED)


if __name__ == "__main__":
    asyncio.run(main())
