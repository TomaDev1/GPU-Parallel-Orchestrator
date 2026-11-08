"""Task graph: tasks, their dependencies, and readiness tracking.

The graph is a DAG. A task becomes *ready* the moment every task it depends on
has succeeded -- not when some "stage" of the graph has finished. Tasks may be
added while a run is in progress (for example, subtasks spawned by an agent);
the scheduler picks them up immediately.
"""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Union

# A prompt is either a fixed string or a function that builds the prompt from
# the outputs of the task's dependencies ({dep_id: output_text}).
PromptSpec = Union[str, Callable[[Mapping[str, str]], str]]


class TaskState(enum.Enum):
    PENDING = "pending"  # waiting on dependencies
    READY = "ready"  # dependencies satisfied, waiting for a slot
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"  # a dependency failed, so this can never run

    @property
    def finished(self) -> bool:
        return self in (TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELLED)


class CycleError(ValueError):
    """Adding a task would create a dependency cycle."""


_seq = itertools.count()


@dataclass
class Task:
    id: str
    prompt: PromptSpec
    deps: tuple[str, ...] = ()
    system: str | None = None
    max_tokens: int = 512
    priority: int = 0  # higher runs first among equally critical tasks
    est_output_tokens: int | None = None  # used for critical-path ordering
    metadata: dict[str, Any] = field(default_factory=dict)

    # Runtime state, managed by the graph/scheduler.
    state: TaskState = field(default=TaskState.PENDING, init=False)
    output: str | None = field(default=None, init=False)
    error: str | None = field(default=None, init=False)
    seq: int = field(default_factory=lambda: next(_seq), init=False)

    def __post_init__(self) -> None:
        self.deps = tuple(self.deps)
        if self.id in self.deps:
            raise CycleError(f"task {self.id!r} depends on itself")

    @property
    def cost(self) -> int:
        """Rough work estimate in output tokens."""
        return self.est_output_tokens if self.est_output_tokens is not None else self.max_tokens

    def build_prompt(self, dep_outputs: Mapping[str, str]) -> str:
        return self.prompt(dep_outputs) if callable(self.prompt) else self.prompt


class TaskGraph:
    def __init__(self, tasks: list[Task] | None = None) -> None:
        self._tasks: dict[str, Task] = {}
        self._dependents: dict[str, set[str]] = {}
        self._listeners: list[Callable[[Task], None]] = []
        for t in tasks or []:
            self.add(t)

    # -- construction -----------------------------------------------------

    def add(self, task: Task) -> Task:
        """Add a task. Dependencies must already be in the graph.

        Requiring dependencies to exist first makes cycles impossible by
        construction for normal use; we still check explicitly.
        """
        if task.id in self._tasks:
            raise ValueError(f"duplicate task id {task.id!r}")
        missing = [d for d in task.deps if d not in self._tasks]
        if missing:
            raise KeyError(f"task {task.id!r} depends on unknown task(s): {missing}")
        self._tasks[task.id] = task
        self._dependents[task.id] = set()
        for d in task.deps:
            self._dependents[d].add(task.id)

        # Initial state given what has already happened to the dependencies.
        dep_states = [self._tasks[d].state for d in task.deps]
        if any(s in (TaskState.FAILED, TaskState.CANCELLED) for s in dep_states):
            task.state = TaskState.CANCELLED
            task.error = "dependency failed"
        elif all(s is TaskState.SUCCEEDED for s in dep_states):
            task.state = TaskState.READY
        for listener in self._listeners:
            listener(task)
        return task

    def on_add(self, listener: Callable[[Task], None]) -> None:
        self._listeners.append(listener)

    def off_add(self, listener: Callable[[Task], None]) -> None:
        self._listeners.remove(listener)

    # -- queries ----------------------------------------------------------

    def __getitem__(self, task_id: str) -> Task:
        return self._tasks[task_id]

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._tasks

    def __iter__(self):
        return iter(self._tasks.values())

    def __len__(self) -> int:
        return len(self._tasks)

    def dependents(self, task_id: str) -> set[str]:
        return set(self._dependents[task_id])

    def dep_outputs(self, task: Task) -> dict[str, str]:
        return {d: self._tasks[d].output or "" for d in task.deps}

    def ready(self) -> list[Task]:
        return [t for t in self._tasks.values() if t.state is TaskState.READY]

    @property
    def done(self) -> bool:
        return all(t.state.finished for t in self._tasks.values())

    def critical_path(self, task_id: str, _memo: dict[str, int] | None = None) -> int:
        """Estimated work on the longest chain starting at this task (inclusive).

        Running tasks with long downstream chains first shortens total
        completion time: the chain is what everything else waits on.
        """
        memo = {} if _memo is None else _memo
        if task_id in memo:
            return memo[task_id]
        # Iterative post-order to avoid recursion limits on deep graphs.
        stack = [(task_id, False)]
        while stack:
            tid, expanded = stack.pop()
            if tid in memo:
                continue
            kids = self._dependents[tid]
            if expanded:
                memo[tid] = self._tasks[tid].cost + max((memo[k] for k in kids), default=0)
            else:
                stack.append((tid, True))
                stack.extend((k, False) for k in kids if k not in memo)
        return memo[task_id]

    # -- state transitions --------------------------------------------------

    def mark_running(self, task: Task) -> None:
        task.state = TaskState.RUNNING

    def mark_succeeded(self, task: Task, output: str) -> list[Task]:
        """Record success; return the tasks that just became ready."""
        task.state = TaskState.SUCCEEDED
        task.output = output
        unlocked = []
        for child_id in self._dependents[task.id]:
            child = self._tasks[child_id]
            if child.state is TaskState.PENDING and all(
                self._tasks[d].state is TaskState.SUCCEEDED for d in child.deps
            ):
                child.state = TaskState.READY
                unlocked.append(child)
        return unlocked

    def mark_failed(self, task: Task, error: str) -> list[Task]:
        """Record failure; cancel everything downstream. Returns cancelled tasks."""
        task.state = TaskState.FAILED
        task.error = error
        cancelled = []
        stack = list(self._dependents[task.id])
        while stack:
            child = self._tasks[stack.pop()]
            if child.state in (TaskState.PENDING, TaskState.READY):
                child.state = TaskState.CANCELLED
                child.error = f"dependency {task.id!r} failed"
                cancelled.append(child)
                stack.extend(self._dependents[child.id])
        return cancelled
