import asyncio

import pytest

from orchestrator import BackendPool, Completion, Endpoint, Request, Scheduler, Task, TaskGraph


class SlowBackend:
    def __init__(self, delay=0.02):
        self.delay = delay
        self.in_flight = 0
        self.peak = 0
        self.keys = []

    async def complete(self, request):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        self.keys.append(request.prefix_key)
        try:
            await asyncio.sleep(self.delay)
            return Completion(text="ok", output_tokens=1)
        finally:
            self.in_flight -= 1


def pool_of(n, strategy="prefix", cap=4, **kw):
    backends = [SlowBackend() for _ in range(n)]
    pool = BackendPool(
        [Endpoint(f"e{i}", b, max_concurrency=cap) for i, b in enumerate(backends)],
        strategy=strategy, **kw,
    )
    return pool, backends


def test_per_endpoint_limits_are_respected():
    pool, backends = pool_of(2, cap=3)

    async def main():
        await asyncio.gather(*(pool.complete(Request(f"r{i}")) for i in range(20)))

    asyncio.run(main())
    assert all(b.peak <= 3 for b in backends)
    assert sum(e.served for e in pool.endpoints) == 20
    assert pool.capacity == 6


def test_prefix_requests_stick_to_one_endpoint_when_balanced():
    pool, backends = pool_of(4, cap=8, max_imbalance=8)

    async def main():
        # Sequential requests: load is always balanced, so affinity decides.
        for _ in range(3):
            for sys_prompt in ("A", "B", "C", "D"):
                await pool.complete(Request("q", system=sys_prompt))

    asyncio.run(main())
    for b in backends:
        assert len(set(b.keys)) == 1  # each endpoint saw exactly one prefix
    assert pool.affinity_hits == 8


def test_prefix_affinity_spills_when_imbalanced():
    pool, backends = pool_of(2, cap=16, max_imbalance=2)

    async def main():
        await asyncio.gather(*(pool.complete(Request("q", system="same")) for _ in range(10)))

    asyncio.run(main())
    # Without spilling, all 10 would land on one endpoint.
    assert all(e.served > 0 for e in pool.endpoints)
    # 10 requests, 2 endpoints, imbalance <= 2 -> at most 6 on one endpoint.
    assert max(b.peak for b in backends) <= 6


def test_round_robin_spreads_evenly():
    pool, _ = pool_of(4, strategy="round_robin")

    async def main():
        for _ in range(8):
            await pool.complete(Request("q", system="same"))

    asyncio.run(main())
    assert [e.served for e in pool.endpoints] == [2, 2, 2, 2]


def test_completion_records_endpoint_in_report():
    pool, _ = pool_of(2)
    g = TaskGraph([Task(f"t{i}", "x") for i in range(4)])
    report = asyncio.run(Scheduler(pool, max_concurrency=pool.capacity).run(g))
    assert {r.endpoint for r in report.records.values()} <= {"e0", "e1"}
    assert all(r.endpoint for r in report.records.values())


def test_bad_config():
    with pytest.raises(ValueError):
        BackendPool([])
    with pytest.raises(ValueError):
        BackendPool([Endpoint("a", SlowBackend())], strategy="nope")
    with pytest.raises(ValueError):
        BackendPool([Endpoint("a", SlowBackend()), Endpoint("a", SlowBackend())])


def test_explicit_prefix_key_overrides_system_hash():
    assert Request("q", system="s", prefix_key="k").prefix_key == "k"
    assert Request("q", system="s").prefix_key == Request("other", system="s").prefix_key
    assert Request("q").prefix_key is None
