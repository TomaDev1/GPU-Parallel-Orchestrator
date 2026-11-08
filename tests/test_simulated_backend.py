import asyncio

import pytest

from orchestrator import Request, Scheduler, SimulatedBackend, Task, TaskGraph


def test_speed_curve_interpolation_and_tail():
    be = SimulatedBackend(speed_curve={1: 100.0, 4: 40.0})
    assert be.speed(1) == 100.0
    assert be.speed(4) == 40.0
    assert be.speed(2) == pytest.approx(80.0)
    # Beyond the last point aggregate throughput stays flat.
    assert be.aggregate_speed(8) == pytest.approx(be.aggregate_speed(4))


def test_single_request_duration_matches_curve():
    be = SimulatedBackend(speed_curve={1: 100.0}, prefill_tps=1e9, time_scale=1000, tick=0.01)

    async def main():
        await be.complete(Request("hi", max_tokens=200))
        return be.now()

    assert asyncio.run(main()) == pytest.approx(2.0, abs=0.05)


def test_batching_beats_sequential_on_wide_graph():
    def graph():
        return TaskGraph([Task(f"t{i}", "x", max_tokens=200) for i in range(8)])

    async def makespan(c):
        be = SimulatedBackend(time_scale=1000, tick=0.02)
        r = await Scheduler(be, max_concurrency=c, clock=be.now).run(graph())
        return r.makespan

    seq, par = asyncio.run(makespan(1)), asyncio.run(makespan(8))
    # Default curve: 123 tok/s alone vs 8 x 45 = 360 tok/s batched.
    assert par < seq / 2


def test_max_batch_queues_extra_requests():
    be = SimulatedBackend(max_batch=2, time_scale=1000)

    async def main():
        await asyncio.gather(*(be.complete(Request("x", max_tokens=20)) for _ in range(5)))

    asyncio.run(main())
    assert be.peak_concurrency == 2


def test_prefix_cache_skips_system_prefill():
    from orchestrator import SimClock

    async def run(cache_size):
        be = SimulatedBackend(prefill_tps=1000, prefix_cache_size=cache_size,
                              clock=SimClock(time_scale=1000, tick=0.01))
        system = "s" * 8000  # 2000 tokens = 2s of prefill
        await be.complete(Request("q", system=system, max_tokens=1))
        second = await be.complete(Request("q", system=system, max_tokens=1))
        return second

    warm = asyncio.run(run(8))
    cold = asyncio.run(run(0))
    assert warm.cached_tokens == 2000 and cold.cached_tokens == 0
    assert warm.ttft < 0.2 < cold.ttft


def test_shared_clock_keeps_backends_in_step():
    from orchestrator import SimClock

    clock = SimClock(time_scale=1000, tick=0.01)
    a = SimulatedBackend(speed_curve={1: 100.0}, prefill_tps=1e9, clock=clock)
    b = SimulatedBackend(speed_curve={1: 100.0}, prefill_tps=1e9, clock=clock)

    async def main():
        await asyncio.gather(a.complete(Request("x", max_tokens=100)),
                             b.complete(Request("x", max_tokens=100)))

    asyncio.run(main())
    # Two separate servers in parallel: both finish in ~1s, not 2s.
    assert clock.now() == pytest.approx(1.0, abs=0.05)
