import asyncio

import pytest

from orchestrator import (
    AdaptiveLimit,
    BackendPool,
    Completion,
    Endpoint,
    Request,
    Scheduler,
    SimClock,
    SimulatedBackend,
    Task,
    TaskGraph,
)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def feed(ctrl, clock, n, speed_per_request, tokens=100, completions=None):
    """Steady state: n requests always in flight, each generating at the given speed.

    Total throughput is n * speed_per_request; a request completes every
    tokens / (n * speed) seconds and is immediately replaced.
    """
    completions = completions or 3 * max(n, ctrl.samples_per_step)
    while ctrl._in_flight < n:
        ctrl.on_start()
    start_limit = ctrl.limit
    for _ in range(completions):
        clock.t += tokens / (n * speed_per_request)
        ctrl.on_complete(tokens)
        if ctrl.limit != start_limit:
            return  # the caller re-feeds at the new limit
        ctrl.on_start()


def drain(ctrl):
    while ctrl._in_flight:
        ctrl.on_complete(0)


def test_measures_total_throughput_at_limit():
    clock = FakeClock()
    ctrl = AdaptiveLimit(initial=4, max_limit=4, samples_per_step=4, clock=clock)
    feed(ctrl, clock, 4, speed_per_request=50.0)
    assert ctrl.curve[4] == pytest.approx(200.0)
    assert ctrl.best == (4, pytest.approx(200.0))


def run_probe(ctrl, clock, speed):
    for _ in range(50):
        if ctrl.phase != "probe":
            return
        n = ctrl.limit
        feed(ctrl, clock, n, speed[n])
        drain(ctrl)


def test_probe_doubles_then_tunes_at_best_level():
    clock = FakeClock()
    ctrl = AdaptiveLimit(initial=2, max_limit=64, samples_per_step=4, clock=clock)
    # Total throughput: 120, 220, 400, 640, 320 -> peaks at 16.
    speed = {2: 60, 4: 55, 8: 50, 16: 40, 32: 10}
    run_probe(ctrl, clock, speed)
    assert ctrl.phase == "tune"
    assert [s.limit for s in ctrl.history][:5] == [2, 4, 8, 16, 32]
    assert ctrl.limit == 16


def test_probe_skips_local_dip():
    # 2 concurrent is worse in total than 1 (like the Strata reference curve),
    # but 4+ is much better. Doubling with a drop tolerance gets past the dip.
    clock = FakeClock()
    ctrl = AdaptiveLimit(initial=1, max_limit=8, samples_per_step=2, clock=clock)
    run_probe(ctrl, clock, {1: 123, 2: 57, 4: 51, 8: 45})
    assert ctrl.best[0] == 8


def test_plateau_prefers_lowest_level():
    ctrl = AdaptiveLimit(initial_curve={8: 300.0, 16: 500.0, 32: 510.0, 64: 505.0})
    assert ctrl.best_level() == 16
    assert ctrl.phase == "tune" and ctrl.limit == 16  # warm start skips probing


def test_warm_start_resumes_probe_if_peak_not_confirmed():
    ctrl = AdaptiveLimit(initial_curve={4: 100.0, 8: 190.0}, max_limit=64)
    assert ctrl.phase == "probe" and ctrl.limit == 8


def test_low_demand_does_not_move_limit():
    clock = FakeClock()
    ctrl = AdaptiveLimit(initial=16, samples_per_step=2, clock=clock)
    feed(ctrl, clock, 2, speed_per_request=50.0, completions=200)  # only 2 in flight
    assert ctrl.limit == 16 and ctrl.history == []


def test_tune_explores_neighbours():
    clock = FakeClock()
    ctrl = AdaptiveLimit(initial=16, max_limit=16, samples_per_step=2, clock=clock)
    run_probe(ctrl, clock, {16: 40})  # probe ends at max_limit -> tune at 16
    assert ctrl.phase == "tune"
    seen = set()
    for _ in range(8):
        seen.add(ctrl.limit)
        feed(ctrl, clock, ctrl.limit, 40)
        drain(ctrl)
    assert seen == {14, 16}  # explores below (above is capped)


def test_bad_bounds():
    with pytest.raises(ValueError):
        AdaptiveLimit(min_limit=5, max_limit=4)


def test_controller_finds_peak_on_simulated_server():
    curve = {1: 46.0, 32: 22.0, 64: 8.0}  # total throughput peaks near 41
    clock = SimClock(time_scale=5000, tick=0.1)
    sim = SimulatedBackend(speed_curve=curve, max_batch=0, clock=clock)
    ctrl = AdaptiveLimit(initial=4, max_limit=64, clock=clock.now)
    pool = BackendPool([Endpoint("gpu0", sim, controller=ctrl)])
    g = TaskGraph([Task(f"t{i}", "x", max_tokens=200) for i in range(1500)])
    asyncio.run(Scheduler(pool, max_concurrency=pool.capacity, clock=clock.now).run(g))
    assert ctrl.phase == "tune"
    assert 24 <= ctrl.best[0] <= 56
    assert pool.capacity == 64


class RecordingBackend:
    def __init__(self):
        self.in_flight = 0
        self.peak = 0

    async def complete(self, request):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(0.02)
            return Completion(text="ok", output_tokens=1)
        finally:
            self.in_flight -= 1


def test_prefill_budget_limits_long_prompts_in_flight():
    be = RecordingBackend()
    pool = BackendPool([Endpoint("e", be, max_concurrency=16, max_prefill_tokens=2500)])
    long_prompt = "x" * 4000  # ~1000 tokens

    async def main():
        await asyncio.gather(*(pool.complete(Request(long_prompt)) for _ in range(10)))

    asyncio.run(main())
    assert be.peak == 2  # 2 x 1000 fits in 2500, a third would not


def test_prefill_budget_admits_one_oversized_request_when_idle():
    be = RecordingBackend()
    pool = BackendPool([Endpoint("e", be, max_prefill_tokens=10)])
    asyncio.run(pool.complete(Request("x" * 4000)))
    assert be.peak == 1


def test_prefill_budget_counts_cached_prefix_as_free():
    be = RecordingBackend()
    pool = BackendPool([Endpoint("e", be, max_concurrency=16, max_prefill_tokens=1200)])
    system = "s" * 4000  # ~1000-token shared prefix

    async def main():
        await pool.complete(Request("q", system=system))  # warms the prefix
        await asyncio.gather(*(pool.complete(Request("q", system=system)) for _ in range(8)))

    asyncio.run(main())
    assert be.peak == 8
