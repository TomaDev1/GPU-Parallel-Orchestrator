"""Subagent API: let a finished task add more work to the graph.

Give a task a ``then`` callback. When the task succeeds, the scheduler calls
``then(task, spawner)``; anything spawned is scheduled immediately.

    def plan_then(task, spawn):
        parts = task.output.splitlines()
        subs = [spawn.task(f"part{i}", f"Do: {p}") for i, p in enumerate(parts)]
        spawn.join("merge", lambda d: "Merge:\\n" + "\\n".join(d.values()), subs)

    graph.add(Task("plan", "List the parts, one per line.", then=plan_then))

Spawned tasks depend on the parent by default, so they can read its output,
and inherit its system prompt (and therefore its prefix-cache key) unless you
override it.
"""

from __future__ import annotations

from typing import Any, Iterable

from .graph import PromptSpec, Task, TaskGraph


class Spawner:
    def __init__(self, graph: TaskGraph, parent: Task) -> None:
        self.graph = graph
        self.parent = parent
        self.spawned: list[Task] = []

    def _qualify(self, task_id: str) -> str:
        # Namespace under the parent so ids from different parents don't clash.
        return task_id if "/" in task_id else f"{self.parent.id}/{task_id}"

    def add(self, task: Task, depend_on_parent: bool = True) -> Task:
        """Add a fully built Task. Its id is used as-is."""
        if depend_on_parent and self.parent.id not in task.deps:
            task.deps = (self.parent.id, *task.deps)
        self.graph.add(task)
        self.spawned.append(task)
        return task

    def task(
        self,
        task_id: str,
        prompt: PromptSpec,
        deps: Iterable[Task | str] = (),
        **kwargs: Any,
    ) -> Task:
        """Spawn a subtask. ``task_id`` is namespaced as ``parent/task_id``.

        Unless given, ``system``, ``prefix_key`` and ``max_tokens`` are
        inherited from the parent.
        """
        for attr in ("system", "prefix_key", "max_tokens"):
            kwargs.setdefault(attr, getattr(self.parent, attr))
        dep_ids = tuple(d.id if isinstance(d, Task) else d for d in deps)
        return self.add(Task(self._qualify(task_id), prompt, deps=dep_ids, **kwargs))

    def join(
        self,
        task_id: str,
        prompt: PromptSpec,
        of: Iterable[Task | str],
        **kwargs: Any,
    ) -> Task:
        """Spawn a task that runs after all of ``of`` (e.g. a merge step)."""
        return self.task(task_id, prompt, deps=of, **kwargs)

    def fan_out(
        self,
        prompts: dict[str, PromptSpec],
        merge_prompt: PromptSpec,
        merge_id: str = "merge",
        **kwargs: Any,
    ) -> Task:
        """Spawn one subtask per prompt plus a merge task; returns the merge task."""
        subs = [self.task(tid, p, **kwargs) for tid, p in prompts.items()]
        return self.join(merge_id, merge_prompt, of=subs, **kwargs)
