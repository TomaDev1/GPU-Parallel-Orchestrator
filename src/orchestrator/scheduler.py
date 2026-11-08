"""Scheduler: keeps as many ready tasks in flight as the concurrency limit allows.

Key behaviours:

* **Dependency-driven, not stage-driven.** A task starts the moment its own
  dependencies succeed. Nothing waits for unrelated tasks to finish.
* **Work-conserving.** Whenever a slot is free and a task is ready, it starts.
* **Critical path first.** Among ready tasks, the one with the most estimated
  work downstream of it goes first, because that chain bounds total time.
* **Dynamic graphs.** Tasks added to the graph during a run (for example by an
  ``on_task_done`` hook spawning subtasks) are scheduled immediately.

The concurrency limit is fixed in this version. An adaptive controller that
tunes it from measured throughput comes later.
"""

from __future__ import annotations

import asyncio
import heapq
import inspect
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .backends import Backend, Request
from .graph import Task, TaskGraph, TaskState
from .spawn import Spawner

TaskHook = Callable[[Task, TaskGraph], "Awaitable[None] | None"]


@dataclass
class TaskRecord:
    task_id: str
    state: TaskState
    ready_at: float | None = None
    started_at: float | None = None
    finished_at: float | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    endpoint: str | None = None
    parent: str | None = None
    deps: tuple[str, ...] = ()
    error: str | None = None

    @property
    def wait(self) -> float | None:
        """Time spent ready but waiting for a slot."""
        if self.ready_at is None or self.started_at is None:
            return None
        return self.started_at - self.ready_at

    @property
    def duration(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.started_at


@dataclass
class RunReport:
    records: dict[str, TaskRecord]
    started_at: float
    finished_at: float
    peak_in_flight: int

    @property
    def makespan(self) -> float:
        """Wall-clock time for the whole graph."""
        return self.finished_at - self.started_at

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records.values())

    @property
    def throughput(self) -> float:
        """Aggregate output tokens per second over the whole run."""
        return self.total_output_tokens / self.makespan if self.makespan > 0 else 0.0

    def critical_path(self) -> list[str]:
        """The chain of tasks that actually determined when the run finished.

        Start from the task that finished last; repeatedly step to whichever
        of its dependencies finished last (the one it was really waiting
        for). Where a task started well after its last dependency finished,
        it was waiting for a free slot rather than for data, and the gap
        shows up as queue wait on that task.
        """
        done = {k: r for k, r in self.records.items() if r.finished_at is not None}
        if not done:
            return []
        current = max(done.values(), key=lambda r: r.finished_at)
        path = [current.task_id]
        while True:
            deps = [done[d] for d in current.deps if d in done]
            if not deps:
                break
            current = max(deps, key=lambda r: r.finished_at)
            path.append(current.task_id)
        return path[::-1]

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "makespan": self.makespan,
            "throughput": self.throughput,
            "peak_in_flight": self.peak_in_flight,
            "critical_path": self.critical_path(),
            "tasks": [
                {
                    "task_id": r.task_id,
                    "state": r.state.value,
                    "parent": r.parent,
                    "deps": list(r.deps),
                    "endpoint": r.endpoint,
                    "ready_at": r.ready_at,
                    "started_at": r.started_at,
                    "finished_at": r.finished_at,
                    "wait": r.wait,
                    "prompt_tokens": r.prompt_tokens,
                    "output_tokens": r.output_tokens,
                    "cached_tokens": r.cached_tokens,
                    "error": r.error,
                }
                for r in self.records.values()
            ],
        }

    def count(self, state: TaskState) -> int:
        return sum(1 for r in self.records.values() if r.state is state)

    def summary(self) -> str:
        waits = [r.wait for r in self.records.values() if r.wait is not None]
        avg_wait = sum(waits) / len(waits) if waits else 0.0
        return (
            f"{len(self.records)} tasks | succeeded {self.count(TaskState.SUCCEEDED)}, "
            f"failed {self.count(TaskState.FAILED)}, cancelled {self.count(TaskState.CANCELLED)} | "
            f"makespan {self.makespan:.2f}s | {self.throughput:.1f} tok/s aggregate | "
            f"peak in flight {self.peak_in_flight} | avg queue wait {avg_wait:.2f}s"
        )


class Scheduler:
    def __init__(
        self,
        backend: Backend,
        max_concurrency: int = 8,
        clock: Callable[[], float] = time.monotonic,
        on_task_done: TaskHook | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.backend = backend
        self.max_concurrency = max_concurrency
        self.clock = clock
        self.on_task_done = on_task_done

    async def run(self, graph: TaskGraph) -> RunReport:
        heap: list[tuple[int, int, int, str]] = []
        records: dict[str, TaskRecord] = {}
        running: dict[asyncio.Task, Task] = {}
        wake = asyncio.Event()
        peak = 0

        def record(task: Task) -> TaskRecord:
            if task.id not in records:
                records[task.id] = TaskRecord(task.id, task.state, deps=tuple(task.deps))
            return records[task.id]

        def push(task: Task) -> None:
            rec = record(task)
            rec.state = task.state
            if task.state is TaskState.READY:
                rec.ready_at = self.clock()
                cp = graph.critical_path(task.id)
                heapq.heappush(heap, (-cp, -task.priority, task.seq, task.id))
            elif task.state is TaskState.CANCELLED:
                rec.error = task.error
            wake.set()

        graph.on_add(push)
        for t in graph:
            push(t) if t.state is TaskState.READY else record(t)

        started_at = self.clock()
        try:
            while True:
                # Fill every free slot with the most critical ready task.
                while heap and len(running) < self.max_concurrency:
                    _, _, _, tid = heapq.heappop(heap)
                    task = graph[tid]
                    if task.state is not TaskState.READY:
                        continue  # cancelled while queued
                    graph.mark_running(task)
                    rec = record(task)
                    rec.state = TaskState.RUNNING
                    rec.started_at = self.clock()
                    running[asyncio.ensure_future(self._execute(graph, task))] = task
                peak = max(peak, len(running))

                if not running:
                    if graph.done or not heap:
                        break
                    continue

                wake.clear()
                waiter = asyncio.ensure_future(wake.wait())
                done, _ = await asyncio.wait(
                    [*running, waiter], return_when=asyncio.FIRST_COMPLETED
                )
                waiter.cancel()
                for fut in done:
                    if fut is waiter:
                        continue
                    task = running.pop(fut)
                    await self._finish(graph, task, fut, record(task), push, record)
        finally:
            graph.off_add(push)
            for fut in running:
                fut.cancel()

        # Anything never reached (e.g. pending on a task that never ran).
        for t in graph:
            rec = record(t)
            rec.state = t.state
        return RunReport(records, started_at, self.clock(), peak)

    async def _execute(self, graph: TaskGraph, task: Task):
        prompt = task.build_prompt(graph.dep_outputs(task))
        return await self.backend.complete(
            Request(
                prompt=prompt,
                system=task.system,
                max_tokens=task.max_tokens,
                task_id=task.id,
                prefix_key=task.prefix_key,
            )
        )

    async def _finish(self, graph, task, fut, rec, push, record) -> None:
        rec.finished_at = self.clock()
        exc = fut.exception()
        if exc is not None:
            rec.state = TaskState.FAILED
            rec.error = f"{type(exc).__name__}: {exc}"
            for c in graph.mark_failed(task, rec.error):
                push(c)
            return
        completion = fut.result()
        rec.prompt_tokens = completion.prompt_tokens
        rec.output_tokens = completion.output_tokens
        rec.cached_tokens = completion.cached_tokens
        rec.endpoint = completion.endpoint
        # Time queued inside a BackendPool counts as waiting, not running.
        # (Assumes the pool and the scheduler use the same clock.)
        if completion.queue_wait and rec.started_at is not None:
            rec.started_at = min(rec.started_at + completion.queue_wait, rec.finished_at)
        rec.state = TaskState.SUCCEEDED
        for child in graph.mark_succeeded(task, completion.text):
            push(child)
        if task.then is not None:
            spawner = Spawner(graph, task)
            result = task.then(task, spawner)
            if inspect.isawaitable(result):
                await result
            for child in spawner.spawned:
                record(child).parent = task.id
        if self.on_task_done is not None:
            result = self.on_task_done(task, graph)
            if inspect.isawaitable(result):
                await result
