# GPU-Parallel-Orchestrator

A coordinator for multi-agent LLM work that keeps a local GPU inference server
as busy as is useful. It breaks a job into a graph of tasks and runs every task
the moment its inputs are ready. It keeps several requests in flight so a
batching server (Strata, vLLM, SGLang, llama.cpp) can process them together.

The goal is the shortest total time to finish the whole job, not the most
agents running at once.

## Status

**Step 1 of 4: core coordinator.** Built and tested against a simulated
batching backend. The target workstation (EPYC + 4 GPUs) isn't built yet.

| Step | What | State |
|---|---|---|
| 1 | Task graph, dependency-driven scheduler, backend interface, simulator | done |
| 2 | Batch-friendly submission: subagent spawning API, shared-prefix grouping | next |
| 3 | Adaptive admission controller (tune concurrency from measured throughput) | planned |
| 4 | Per-task metrics store and timeline view | planned |

## How it works

```
objective ──> TaskGraph (DAG) ──> Scheduler ──> Backend ──> inference server ──> GPUs
                  ▲                    │
                  └── new subtasks ────┘
```

- **TaskGraph** holds tasks and their dependencies. A task's prompt can be a
  string or a function of its dependencies' outputs. Tasks can be added
  mid-run, for example when an agent spawns subtasks.
- **Scheduler** starts a ready task whenever a slot is free, up to
  `max_concurrency`. It starts each task as soon as *its own* dependencies
  finish, not when a whole stage does. Among ready tasks it picks the one with
  the longest estimated chain of work downstream (the critical path). If a
  task fails, only the tasks that depend on it are cancelled.
- **Backends** expose `async complete(request)`:
  - `OpenAICompatibleBackend` calls `/chat/completions` on any compatible server.
  - `SimulatedBackend` is a fake continuous-batching server for development
    without GPUs. Per-request speed drops as more requests share the batch,
    following a configurable speed curve.

## Quick start

```bash
pip install -e .[dev]
python -m pytest
python examples/demo_simulated.py      # no GPU needed
python examples/demo_server.py --base-url http://localhost:8000/v1 --model <name>
```

```python
from orchestrator import Scheduler, SimulatedBackend, Task, TaskGraph

g = TaskGraph()
g.add(Task("research", "Research X.", max_tokens=800))
g.add(Task("cpu", lambda d: f"Evaluate CPUs given: {d['research']}", deps=("research",)))
g.add(Task("gpu", lambda d: f"Evaluate GPUs given: {d['research']}", deps=("research",)))
g.add(Task("compare", "Compare.", deps=("cpu", "gpu")))

backend = SimulatedBackend(time_scale=100)
report = await Scheduler(backend, max_concurrency=8, clock=backend.now).run(g)
print(report.summary())
```

## Simulator results (illustrative only)

`examples/demo_simulated.py` with the default speed curve (123 tok/s for one
request, 57 each at 2, 51 at 4, 45 at 8). That curve comes from a published
Strata 4-GPU reference run and **has not been verified for this hardware**.

| Graph | concurrency 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Architecture review (9 tasks, deep chain) | 45.5s | 47.9s | 39.3s | 35.9s | 35.9s |
| Plan → 16 subtasks → merge | 61.1s | 64.3s | 39.1s | 25.4s | 25.3s |

Two takeaways for later steps:

1. With this curve, **2 concurrent requests is slower than 1** (2 × 57 < 123),
   and the gains only show up at 4 and above. A fixed concurrency limit is easy
   to get wrong, which is why step 3 measures and adapts.
2. A deep graph is limited by its critical path, so more concurrency stops
   helping early. Decomposing work into genuinely parallel subtasks matters
   as much as the scheduler.
