import asyncio

from orchestrator import (
    AdaptiveLimit,
    BackendPool,
    Completion,
    Endpoint,
    MetricsStore,
    Scheduler,
    Task,
    TaskGraph,
    render_timeline,
    write_timeline,
)


class DelayBackend:
    def __init__(self, delays):
        self.delays = delays

    async def complete(self, request):
        await asyncio.sleep(self.delays.get(request.task_id, 0.01))
        return Completion(text="ok", output_tokens=10)


def diamond_report():
    # a -> (fast, slow) -> end : critical path goes through `slow`.
    g = TaskGraph([
        Task("a", "x"),
        Task("fast", "x", deps=("a",)),
        Task("slow", "x", deps=("a",)),
        Task("end", "x", deps=("fast", "slow")),
    ])
    be = DelayBackend({"a": 0.02, "fast": 0.01, "slow": 0.08, "end": 0.02})
    return asyncio.run(Scheduler(be).run(g))


def test_critical_path_follows_latest_dependency():
    assert diamond_report().critical_path() == ["a", "slow", "end"]


def test_critical_path_includes_spawned_tasks():
    def then(task, spawn):
        subs = [spawn.task("s0", "x"), spawn.task("s1", "x")]
        spawn.join("merge", "m", of=subs)

    g = TaskGraph([Task("plan", "p", then=then)])
    report = asyncio.run(Scheduler(DelayBackend({"plan/s1": 0.05})).run(g))
    assert report.critical_path() == ["plan", "plan/s1", "plan/merge"]


def test_store_round_trip_and_warm_controller(tmp_path):
    db = tmp_path / "m.db"
    ctrl = AdaptiveLimit(initial=2, initial_curve={2: 50.0, 4: 90.0, 8: 85.0})
    pool = BackendPool([Endpoint("gpu0", DelayBackend({}), controller=ctrl)])
    g = TaskGraph([Task(f"t{i}", "x") for i in range(5)])
    report = asyncio.run(Scheduler(pool, max_concurrency=8).run(g))

    with MetricsStore(db) as store:
        run_id = store.save_run(report, pool, label="test")
        assert store.runs()[0]["id"] == run_id
        assert store.runs()[0]["label"] == "test"
        tasks = store.run_tasks(run_id)
        assert {t["task_id"] for t in tasks} == {f"t{i}" for i in range(5)}
        assert all(t["endpoint"] == "gpu0" for t in tasks)
        assert store.load_curve("gpu0") == {2: 50.0, 4: 90.0, 8: 85.0}

    with MetricsStore(db) as store:  # reopen: data persisted
        warm = store.controller("gpu0", max_limit=64)
        assert warm.phase == "tune" and warm.limit == 4
        cold = store.controller("gpu9")
        assert cold.phase == "probe" and cold.curve == {}


def test_pool_queue_wait_counts_as_waiting():
    # One slot, three tasks: later tasks wait in the pool, not "run" longer.
    pool = BackendPool([Endpoint("e", DelayBackend({}), max_concurrency=1)])
    g = TaskGraph([Task(f"t{i}", "x") for i in range(3)])
    report = asyncio.run(Scheduler(pool, max_concurrency=8).run(g))
    waits = sorted(r.wait for r in report.records.values())
    assert waits[0] < 0.005 and waits[-1] > 0.015
    assert all(r.duration < 0.03 for r in report.records.values())


def test_timeline_renders_self_contained_html(tmp_path):
    report = diamond_report()
    page = render_timeline(report, title="T <test>")
    assert page.startswith("<!doctype html>")
    assert "T &lt;test&gt;" in page
    assert page.count('class="bar crit"') == 3  # a, slow, end
    assert "<script" not in page and "http" not in page
    out = write_timeline(report, tmp_path / "t.html")
    assert out.read_text().startswith("<!doctype html>")
