import asyncio

import pytest

from orchestrator import Completion, Scheduler, SimulatedBackend, Task, TaskGraph, TaskState


class FakeBackend:
    """Each request sleeps for delays[task_id] (default 0.01s) and tracks concurrency."""

    def __init__(self, delays=None, fail=()):
        self.delays = delays or {}
        self.fail = set(fail)
        self.in_flight = 0
        self.peak = 0
        self.started = []
        self.prompts = {}

    async def complete(self, request):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        self.started.append(request.task_id)
        self.prompts[request.task_id] = request.prompt
        try:
            await asyncio.sleep(self.delays.get(request.task_id, 0.01))
            if request.task_id in self.fail:
                raise RuntimeError("model error")
            return Completion(text=f"out:{request.task_id}", output_tokens=10)
        finally:
            self.in_flight -= 1


def run(coro):
    return asyncio.run(coro)


def test_independent_tasks_run_concurrently_up_to_limit():
    g = TaskGraph([Task(f"t{i}", "x") for i in range(10)])
    be = FakeBackend({f"t{i}": 0.05 for i in range(10)})
    report = run(Scheduler(be, max_concurrency=4).run(g))
    assert be.peak == 4
    assert report.count(TaskState.SUCCEEDED) == 10
    assert report.peak_in_flight == 4


def test_task_starts_when_its_own_deps_finish_not_when_stage_finishes():
    # fast -> child should start long before slow (an unrelated sibling) finishes.
    g = TaskGraph([
        Task("fast", "x"), Task("slow", "x"), Task("child", "x", deps=("fast",)),
    ])
    be = FakeBackend({"fast": 0.01, "slow": 0.3, "child": 0.01})
    report = run(Scheduler(be, max_concurrency=4).run(g))
    assert report.records["child"].finished_at < report.records["slow"].finished_at


def test_dependency_outputs_flow_into_prompts():
    g = TaskGraph([Task("a", "x"), Task("b", lambda d: f"got {d['a']}", deps=("a",))])
    be = FakeBackend()
    run(Scheduler(be).run(g))
    assert be.prompts["b"] == "got out:a"
    assert g["b"].output == "out:b"


def test_critical_path_task_is_started_first():
    g = TaskGraph([
        Task("leaf", "x", max_tokens=10),
        Task("head", "x", max_tokens=10),
        Task("tail", "x", deps=("head",), max_tokens=1000),
    ])
    be = FakeBackend()
    run(Scheduler(be, max_concurrency=1).run(g))
    assert be.started[0] == "head"


def test_failure_cancels_dependents_and_others_still_finish():
    g = TaskGraph([
        Task("bad", "x"), Task("after_bad", "x", deps=("bad",)), Task("ok", "x"),
    ])
    report = run(Scheduler(FakeBackend(fail={"bad"})).run(g))
    assert report.records["bad"].state is TaskState.FAILED
    assert "model error" in report.records["bad"].error
    assert report.records["after_bad"].state is TaskState.CANCELLED
    assert report.records["ok"].state is TaskState.SUCCEEDED


def test_hook_can_spawn_subtasks_mid_run():
    g = TaskGraph([Task("plan", "x")])

    def spawn(task, graph):
        if task.id == "plan":
            for i in range(3):
                graph.add(Task(f"sub{i}", "x", deps=("plan",)))
            graph.add(Task("merge", "x", deps=("sub0", "sub1", "sub2")))

    be = FakeBackend({f"sub{i}": 0.05 for i in range(3)})
    report = run(Scheduler(be, max_concurrency=8, on_task_done=spawn).run(g))
    assert report.count(TaskState.SUCCEEDED) == 5
    assert be.peak == 3
    assert be.started[-1] == "merge"


def test_external_add_during_run_is_picked_up():
    g = TaskGraph([Task("a", "x")])
    be = FakeBackend({"a": 0.1})

    async def main():
        sched_task = asyncio.ensure_future(Scheduler(be).run(g))
        await asyncio.sleep(0.02)
        g.add(Task("late", "x"))
        return await sched_task

    report = run(main())
    assert report.records["late"].state is TaskState.SUCCEEDED
    assert be.peak == 2


def test_invalid_concurrency():
    with pytest.raises(ValueError):
        Scheduler(FakeBackend(), max_concurrency=0)


def test_empty_graph():
    report = run(Scheduler(FakeBackend()).run(TaskGraph()))
    assert report.records == {}
