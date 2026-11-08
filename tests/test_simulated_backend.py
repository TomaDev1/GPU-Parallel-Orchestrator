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
