"""Benchmark the orchestrator against real servers (one per GPU), or a simulation.

On a rented multi-GPU machine (see README, "Testing on Vast.ai"):

    ./scripts/launch_vllm.sh                      # one vLLM server per GPU
    python scripts/bench.py                       # adaptive, two runs
    python scripts/bench.py --limits 4,8,16,32    # fixed limits first, then adaptive

Without GPUs (checks the script end to end on the 27B/3080 estimate):

    python scripts/bench.py --simulate

Workload: agent "teams". Each team has its own long shared brief (a cacheable
system prompt), plans, fans out into subtasks, then reviews and finalises.
Results go to a SQLite file and an HTML timeline per run, and a summary
table is printed at the end.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchestrator import (  # noqa: E402
    QWEN27B_RTX3080_20GB,
    BackendPool,
    Endpoint,
    MetricsStore,
    OpenAICompatibleBackend,
    Scheduler,
    SimClock,
    SimulatedBackend,
    Task,
    TaskGraph,
    write_timeline,
)

TOPICS = [
    "power delivery for a 4-GPU workstation", "PCIe lane allocation on EPYC", "cooling dense GPU builds",
    "NUMA effects on inference", "KV cache sizing", "prefix caching in vLLM", "quantisation trade-offs",
    "speculative decoding", "MoE expert offloading", "agent task decomposition", "evaluating LLM outputs",
    "scheduling under memory pressure", "batching and memory bandwidth", "storage for model weights",
    "monitoring GPUs with NVML", "failure handling in agent pipelines",
]

BRIEF_PARAGRAPH = (
    "This team is part of an engineering group designing a local LLM inference workstation built "
    "around an AMD EPYC processor and four consumer GPUs. Every answer must be specific, technically "
    "accurate, and state its assumptions. Prefer numbers with units, name trade-offs explicitly, and "
    "flag anything that should be measured rather than assumed. "
)


def brief(team: int, rng: random.Random) -> str:
    """A long shared system prompt (cacheable prefix), unique per team."""
    reps = rng.randint(12, 40)  # roughly 1k-3k tokens
    return f"Team {team} brief. Focus area: {TOPICS[team % len(TOPICS)]}.\n\n" + BRIEF_PARAGRAPH * reps


def build_graph(teams: int, seed: int, scale: float = 1.0) -> TaskGraph:
    rng = random.Random(seed)
    g = TaskGraph()
    for t in range(teams):
        topic = TOPICS[t % len(TOPICS)]
        n_parts = rng.randint(4, 10)

        def plan_then(task, spawn, topic=topic, n_parts=n_parts):
            subs = [
                spawn.task(
                    f"part{i}",
                    f"Write section {i + 1} of {n_parts} of a technical note on {topic}. "
                    f"Cover a different aspect from the other sections. Use about {int(250 * scale)} words.",
                    max_tokens=int(rng.randint(250, 600) * scale),
                )
                for i in range(n_parts)
            ]
            review = spawn.join(
                "review",
                lambda d: "Review these sections for errors and gaps:\n\n" + "\n\n".join(d.values()),
                of=subs,
                max_tokens=int(500 * scale),
            )
            spawn.join(
                "final",
                lambda d: "Write a short final summary using this review:\n\n" + "\n\n".join(d.values()),
                of=[review],
                max_tokens=int(400 * scale),
            )

        g.add(Task(
            f"team{t}",
            f"Plan a {n_parts}-section technical note on {topic}. List the section titles.",
            system=brief(t, rng),
            max_tokens=int(200 * scale),
            then=plan_then,
        ))
    return g


def endpoint_urls(args) -> list[str]:
    if args.endpoints:
        return [u.strip() for u in args.endpoints.split(",") if u.strip()]
    return [f"http://{args.host}:{args.base_port + i}/v1" for i in range(args.gpus)]


async def detect_model(url: str, api_key: str | None) -> str:
    import httpx

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url.rstrip("/") + "/models", headers=headers)
        r.raise_for_status()
        return r.json()["data"][0]["id"]


async def one_run(args, store: MetricsStore, label: str, fixed: int | None, seed: int, outdir: Path,
                  explore: bool = True) -> dict:
    sim_clock = None
    endpoints = []
    if args.simulate:
        sim_clock = SimClock(time_scale=args.time_scale, tick=0.1)
        backends = [
            (f"sim{i}", SimulatedBackend(clock=sim_clock, **QWEN27B_RTX3080_20GB)) for i in range(args.gpus)
        ]
        clock = sim_clock.now
    else:
        urls = endpoint_urls(args)
        model = args.model or await detect_model(urls[0], args.api_key)
        backends = [
            (f"{args.name_prefix}{i}", OpenAICompatibleBackend(u, model, api_key=args.api_key,
                                                               temperature=args.temperature))
            for i, u in enumerate(urls)
        ]
        clock = time.monotonic

    for name, backend in backends:
        if fixed is not None:
            endpoints.append(Endpoint(name, backend, max_concurrency=fixed,
                                      max_prefill_tokens=args.prefill_budget))
        else:
            ctrl = store.controller(name, initial=4, max_limit=args.max_limit, clock=clock, explore=explore)
            endpoints.append(Endpoint(name, backend, controller=ctrl, max_prefill_tokens=args.prefill_budget))

    pool = BackendPool(endpoints, clock=clock)
    graph = build_graph(args.teams, seed, scale=args.scale)
    try:
        report = await Scheduler(pool, max_concurrency=pool.capacity, clock=clock).run(graph)
    finally:
        await pool.aclose()

    run_id = store.save_run(report, pool if fixed is None else None, label=label)
    timeline = write_timeline(report, outdir / f"timeline-run{run_id}.html", title=f"Run {run_id}: {label}")
    limits = {
        e.name: e.controller.best[0] if e.controller and e.controller.best else e.limit
        for e in pool.endpoints
    }
    failed = [r for r in report.records.values() if r.error and r.state.value == "failed"]
    if failed:
        print(f"  {len(failed)} task(s) failed, e.g. {failed[0].task_id}: {failed[0].error}")
    row = {
        "run": run_id, "label": label, "tasks": len(report.records), "makespan_s": round(report.makespan, 1),
        "tok_per_s": round(report.throughput, 1), "peak_in_flight": report.peak_in_flight,
        "limits": limits, "timeline": str(timeline),
    }
    print(f"  run {run_id} {label:<16} {report.summary()}")
    return row


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", type=int, default=4, help="number of servers (one per GPU)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--base-port", type=int, default=8000)
    ap.add_argument("--endpoints", help="comma-separated base URLs instead of host/ports, e.g. http://a:8000/v1")
    ap.add_argument("--model", help="model name to send (default: first model the server lists)")
    ap.add_argument("--api-key")
    ap.add_argument("--teams", type=int, default=24, help="agent teams per run")
    ap.add_argument("--seed", type=int, default=0, help="workload seed (same for every run, so runs compare fairly)")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply output lengths")
    ap.add_argument("--limits", default="", help="fixed per-GPU limits to try first, e.g. 4,8,16,32")
    ap.add_argument("--adaptive-runs", type=int, default=3, help="adaptive runs (first learns, later ones warm)")
    ap.add_argument("--max-limit", type=int, default=64)
    ap.add_argument("--prefill-budget", type=int, default=16_000, help="uncached prompt tokens in flight per GPU")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--name-prefix", default="gpu", help="endpoint names; curves are saved per name")
    ap.add_argument("--db", default="bench.db")
    ap.add_argument("--out", default="bench-results")
    ap.add_argument("--simulate", action="store_true", help="use simulated cards instead of servers")
    ap.add_argument("--time-scale", type=float, default=2000, help="simulation speed-up")
    args = ap.parse_args()

    if args.simulate:
        args.name_prefix = "sim"
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    fixed_limits = [int(x) for x in args.limits.split(",") if x.strip()]

    rows = []
    with MetricsStore(args.db) as store:
        where = "simulated cards" if args.simulate else ", ".join(endpoint_urls(args))
        print(f"benchmarking {args.teams} agent teams per run on {where}\n")
        for limit in fixed_limits:
            rows.append(await one_run(args, store, f"fixed {limit}", limit, args.seed, outdir))
        for i in range(args.adaptive_runs):
            rows.append(await one_run(args, store, f"adaptive #{i + 1}", None, args.seed, outdir))
        if args.adaptive_runs:
            # Hold each GPU at the best level it has learned, without exploring.
            rows.append(await one_run(args, store, "learned, frozen", None, args.seed, outdir, explore=False))
        curves = {e: store.load_curve(e) for e in {n for r in rows for n in r["limits"]}}

    (outdir / "summary.json").write_text(json.dumps({"runs": rows, "curves": curves}, indent=2))
    print("\nrun  label             makespan   tok/s   per-GPU limit")
    for r in rows:
        print(f"{r['run']:>3}  {r['label']:<16} {r['makespan_s']:>7}s {r['tok_per_s']:>8}   {r['limits']}")
    print("\nlearned curves (concurrency: total output tok/s):")
    for name, curve in sorted(curves.items()):
        if curve:
            print(f"  {name}: " + ", ".join(f"{k}: {v:.0f}" for k, v in sorted(curve.items())))
    print(f"\nresults in {outdir}/ (summary.json, timeline-run*.html) and {args.db}")
    print("Every run uses the same workload (--seed), so makespans compare directly. Real servers add "
          "noise; repeat a run if two results are close.")


if __name__ == "__main__":
    asyncio.run(main())
