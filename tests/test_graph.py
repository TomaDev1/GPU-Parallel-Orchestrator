import pytest

from orchestrator import CycleError, Task, TaskGraph, TaskState


def test_ready_only_when_all_deps_succeed():
    g = TaskGraph([Task("a", "x"), Task("b", "x"), Task("c", "x", deps=("a", "b"))])
    assert {t.id for t in g.ready()} == {"a", "b"}
    assert g.mark_succeeded(g["a"], "A") == []
    assert [t.id for t in g.mark_succeeded(g["b"], "B")] == ["c"]
    assert g["c"].state is TaskState.READY
    assert g.dep_outputs(g["c"]) == {"a": "A", "b": "B"}


def test_failure_cancels_everything_downstream_only():
    g = TaskGraph([
        Task("a", "x"), Task("b", "x", deps=("a",)), Task("c", "x", deps=("b",)), Task("d", "x"),
    ])
    cancelled = g.mark_failed(g["a"], "boom")
    assert {t.id for t in cancelled} == {"b", "c"}
    assert g["d"].state is TaskState.READY


def test_unknown_dep_and_duplicates_rejected():
    g = TaskGraph([Task("a", "x")])
    with pytest.raises(KeyError):
        g.add(Task("b", "x", deps=("nope",)))
    with pytest.raises(ValueError):
        g.add(Task("a", "x"))
    with pytest.raises(CycleError):
        Task("s", "x", deps=("s",))


def test_task_added_after_deps_finished_is_ready_or_cancelled():
    g = TaskGraph([Task("a", "x"), Task("f", "x")])
    g.mark_succeeded(g["a"], "A")
    g.mark_failed(g["f"], "boom")
    assert g.add(Task("b", "x", deps=("a",))).state is TaskState.READY
    assert g.add(Task("c", "x", deps=("f",))).state is TaskState.CANCELLED


def test_critical_path_counts_longest_downstream_chain():
    g = TaskGraph([
        Task("a", "x", max_tokens=10),
        Task("short", "x", deps=("a",), max_tokens=5),
        Task("long1", "x", deps=("a",), max_tokens=100),
        Task("long2", "x", deps=("long1",), max_tokens=100),
    ])
    assert g.critical_path("a") == 210
    assert g.critical_path("short") == 5


def test_prompt_built_from_dependency_outputs():
    t = Task("b", lambda deps: f"use {deps['a']}", deps=())
    assert t.build_prompt({"a": "A"}) == "use A"
