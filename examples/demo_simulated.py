"""Compare sequential vs parallel execution of a task graph on the simulated backend.

    python examples/demo_simulated.py

No GPU or server needed. Times are simulated seconds based on the speed
curve in SimulatedBackend (an unverified reference curve -- replace with
real measurements once the workstation is running).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchestrator import Scheduler, SimulatedBackend, Task, TaskGraph  # noqa: E402


def build_graph() -> TaskGraph:
    """The 'evaluate a new server architecture' example, widened a little."""
    g = TaskGraph()
    g.add(Task("research_hardware", "Research current workstation hardware options.", max_tokens=800))
    g.add(Task("research_software", "Research inference servers for MoE models.", max_tokens=800))
    for part in ("cpu", "gpu", "memory", "storage"):
        g.add(
            Task(
                f"eval_{part}",
                lambda deps, part=part: f"Evaluate {part} options given:\n{deps['research_hardware']}",
                deps=("research_hardware",),
                max_tokens=500,
            )
        )
    g.add(Task("eval_servers", "Compare vLLM, SGLang and Strata.", deps=("research_software",), max_tokens=600))
    g.add(Task("power", "Calculate power requirements.", deps=("eval_cpu", "eval_gpu"), max_tokens=300))
    g.add(
        Task(
            "compare",
            "Compare full architectures and recommend one.",
            deps=("eval_cpu", "eval_gpu", "eval_memory", "eval_storage", "eval_servers", "power"),
            max_tokens=1000,
        )
    )
    return g


def build_fanout(n: int = 16) -> TaskGraph:
    """Plan -> n independent subtasks -> merge. The shape agents produce most."""
    g = TaskGraph()
    g.add(Task("plan", "Split the job into subtasks.", max_tokens=300))
    for i in range(n):
        g.add(Task(f"sub_{i}", f"Do subtask {i}.", deps=("plan",), max_tokens=400))
    g.add(Task("merge", "Merge the results.", deps=tuple(f"sub_{i}" for i in range(n)), max_tokens=600))
    return g


async def run(name: str, build, concurrency: int) -> None:
    backend = SimulatedBackend(time_scale=200)
    report = await Scheduler(backend, max_concurrency=concurrency, clock=backend.now).run(build())
    print(f"{name:<12} concurrency={concurrency:>2}: {report.summary()}")


async def main() -> None:
    for name, build in (("architecture", build_graph), ("fan-out x16", build_fanout)):
        for c in (1, 2, 4, 8, 16):
            await run(name, build, c)
        print()


if __name__ == "__main__":
    asyncio.run(main())
