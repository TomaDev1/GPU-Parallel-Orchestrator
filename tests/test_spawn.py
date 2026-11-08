import asyncio

from orchestrator import Completion, Scheduler, Task, TaskGraph, TaskState


class EchoBackend:
    def __init__(self):
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        await asyncio.sleep(0.01)
        if request.task_id == "plan":
            return Completion(text="alpha\nbeta\ngamma", output_tokens=3)
        return Completion(text=f"done:{request.task_id}", output_tokens=1)


def test_then_spawns_subtasks_and_merge_from_output():
    def plan_then(task, spawn):
        parts = task.output.splitlines()
        subs = [spawn.task(p, lambda d, p=p: f"do {p}") for p in parts]
        spawn.join("merge", lambda d: "|".join(sorted(d)), of=subs)

    g = TaskGraph([Task("plan", "list parts", system="SYS", then=plan_then)])
    be = EchoBackend()
    report = asyncio.run(Scheduler(be).run(g))

    assert report.count(TaskState.SUCCEEDED) == 5
    assert set(report.records) == {"plan", "plan/alpha", "plan/beta", "plan/gamma", "plan/merge"}
    assert report.records["plan/alpha"].parent == "plan"
    by_id = {r.task_id: r for r in be.requests}
    # Subtasks inherit the parent's system prompt, so they share its prefix key.
    assert by_id["plan/beta"].system == "SYS"
    assert by_id["plan/beta"].prefix_key == by_id["plan"].prefix_key
    # Merge sees the parent plus its declared deps.
    assert by_id["plan/merge"].prompt == "plan|plan/alpha|plan/beta|plan/gamma"
    assert be.requests[-1].task_id == "plan/merge"


def test_fan_out_helper_and_parent_dependency():
    def then(task, spawn):
        merge = spawn.fan_out({"a": "A", "b": "B"}, merge_prompt="M", max_tokens=7)
        assert set(merge.deps) == {"root", "root/a", "root/b"}

    g = TaskGraph([Task("root", "r", then=then)])
    be = EchoBackend()
    asyncio.run(Scheduler(be).run(g))
    assert g["root/a"].deps == ("root",)
    assert g["root/merge"].max_tokens == 7
    assert g["root/merge"].state is TaskState.SUCCEEDED


def test_async_then_and_nested_spawning():
    async def level2(task, spawn):
        spawn.task("leaf", "leaf")

    async def level1(task, spawn):
        await asyncio.sleep(0)
        spawn.task("child", "c", then=level2)

    g = TaskGraph([Task("root", "r", then=level1)])
    report = asyncio.run(Scheduler(EchoBackend()).run(g))
    assert "root/child/leaf" in report.records
    assert report.records["root/child/leaf"].state is TaskState.SUCCEEDED
    assert report.records["root/child/leaf"].parent == "root/child"
