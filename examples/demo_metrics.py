"""Run the same job twice with a metrics store, then write an HTML timeline.

    python examples/demo_metrics.py [--db metrics.db] [--out timeline.html]

The first run starts with nothing known about the cards, so the adaptive
controllers probe. Learned speed curves are saved to SQLite, and the second
run starts warm from them. Uses the QWEN27B_RTX3080_20GB simulator estimate
(not real measurements).
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchestrator import (  # noqa: E402
    QWEN27B_RTX3080_20GB,
    BackendPool,
    Endpoint,
    MetricsStore,
    Scheduler,
    SimClock,
    SimulatedBackend,
    Task,
    TaskGraph,
    write_timeline,
)


def build_graph(seed: int) -> TaskGraph:
    """24 agent teams: plan -> 4-12 subtasks -> review -> final, sharing a system prompt."""
    rng = random.Random(seed)
    g = TaskGraph()
    for t in range(24):
        system = f"[team {t} brief] " + "context " * rng.randint(300, 900)
        n = rng.randint(4, 12)

        def plan_then(task, spawn, n=n):
            subs = [
                spawn.task(f"part{i}", f"Work on part {i}.", max_tokens=rng.randint(150, 500))
                for i in range(n)
            ]
            review = spawn.join("review", "Review all parts.", of=subs, max_tokens=400)
            spawn.join("final", "Write the final answer.", of=[review], max_tokens=600)

        g.add(Task(f"team{t}", "Plan the work.", system=system, max_tokens=200, then=plan_then))
    return g


async def run_once(store: MetricsStore, label: str, seed: int):
    clock = SimClock(time_scale=2000, tick=0.1)
    pool = BackendPool([
        Endpoint(
            f"gpu{i}",
            SimulatedBackend(clock=clock, **QWEN27B_RTX3080_20GB),
            controller=store.controller(f"gpu{i}", initial=4, max_limit=64, clock=clock.now),
            max_prefill_tokens=12_000,
        )
        for i in range(4)
    ], clock=clock.now)
    report = await Scheduler(pool, max_concurrency=pool.capacity, clock=clock.now).run(build_graph(seed))
    run_id = store.save_run(report, pool, label=label)
    warm = "warm" if pool.endpoints[0].controller.initial_curve else "cold"
    print(f"run {run_id} ({label}, controllers {warm}): {report.summary()}")
    return report


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", help="SQLite file (default: a temporary file)")
    ap.add_argument("--out", default="timeline.html")
    args = ap.parse_args()
    db = args.db or str(Path(tempfile.mkdtemp()) / "metrics.db")

    with MetricsStore(db) as store:
        await run_once(store, "first run", seed=1)
        report = await run_once(store, "second run", seed=2)
        print(f"\nlearned curve for gpu0: {store.load_curve('gpu0')}")
    path = write_timeline(report, args.out, title="Second run: 24 agent teams on 4 cards (simulated)")
    print(f"metrics in {db}\ntimeline written to {path}")


if __name__ == "__main__":
    asyncio.run(main())
