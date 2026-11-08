"""Four model copies (one per GPU) behind a BackendPool, compared by routing strategy.

    python examples/demo_multi_gpu.py

Workload: 8 agent "teams", each with its own long shared system prompt
(~6k tokens of instructions and context). Each team starts with a planning
task whose ``then`` callback spawns 8 subtasks and a merge step, using the
subagent API.

Each simulated card:
* prefills prompts at 1,800 tok/s (prefill doesn't speed up with batching),
* decodes at 46 tok/s for one request, falling to ~16 tok/s each at 64,
* has room in KV memory for only 2 long prefixes at a time.

Those per-card figures come from third-party RTX 3090 reports for a
different model and are NOT measurements of this hardware. They are here to
show the effect of routing, not to predict real speed.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchestrator import (  # noqa: E402
    BackendPool,
    Endpoint,
    Scheduler,
    SimClock,
    SimulatedBackend,
    Task,
    TaskGraph,
)

PER_CARD_CURVE = {1: 46.0, 64: 16.2}
TEAMS = 8
SUBTASKS = 8
SYSTEM_CHARS = 24_000  # ~6k tokens at 4 chars/token


def build_graph() -> TaskGraph:
    g = TaskGraph()
    for t in range(TEAMS):
        system = f"[team {t} instructions and shared context] " + "x" * SYSTEM_CHARS

        def plan_then(task, spawn):
            spawn.fan_out(
                {f"sub{i}": f"Work on part {i}." for i in range(SUBTASKS)},
                merge_prompt="Merge the parts.",
                max_tokens=300,
            )

        g.add(Task(f"team{t}", "Plan the work.", system=system, max_tokens=150, then=plan_then))
    return g


async def run(strategy: str) -> None:
    clock = SimClock(time_scale=400, tick=0.05)
    sims = [
        SimulatedBackend(
            speed_curve=PER_CARD_CURVE, prefill_tps=1800, max_batch=64,
            prefix_cache_size=2, clock=clock,
        )
        for _ in range(4)
    ]
    pool = BackendPool(
        [Endpoint(f"gpu{i}", s, max_concurrency=16) for i, s in enumerate(sims)],
        strategy=strategy,
    )
    report = await Scheduler(pool, max_concurrency=pool.capacity, clock=clock.now).run(build_graph())
    hits = sum(s.cache_hits for s in sims)
    misses = sum(s.cache_misses for s in sims)
    print(
        f"{strategy:<13} makespan {report.makespan:6.1f}s | "
        f"{report.throughput:6.1f} tok/s | prefix cache hits {hits}/{hits + misses} | "
        f"per card {[e.served for e in pool.endpoints]}"
    )


async def main() -> None:
    print(f"{TEAMS} teams x ({SUBTASKS} subtasks + plan + merge) = {TEAMS * (SUBTASKS + 2)} tasks on 4 cards\n")
    for strategy in ("round_robin", "least_loaded", "prefix"):
        await run(strategy)


if __name__ == "__main__":
    asyncio.run(main())
