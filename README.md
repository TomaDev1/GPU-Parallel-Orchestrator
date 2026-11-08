# GPU-Parallel-Orchestrator

A coordinator for multi-agent LLM work that keeps a local GPU inference server
as busy as is useful. It breaks a job into a graph of tasks and runs every task
the moment its inputs are ready. It keeps several requests in flight so a
batching server (Strata, vLLM, SGLang, llama.cpp) can process them together.

The goal is the shortest total time to finish the whole job, not the most
agents running at once.

## Status

**Step 3 of 4 done.** Built and tested against simulated servers. The target
setup is an EPYC workstation with 4× RTX 3080 20GB, each running its own copy
of Qwen3.8-27B at 4-bit. It isn't built yet.

| Step | What | State |
|---|---|---|
| 1 | Task graph, dependency-driven scheduler, backend interface, simulator | done |
| 2 | Multiple servers (one per GPU), prefix-aware routing, subagent spawning API | done |
| 3 | Adaptive admission controller: tune concurrency per server from measured throughput; separate prompt-token budget | done |
| 4 | Per-task metrics store and timeline view; save learned speed curves between runs | next |

## How it works

```
objective ──> TaskGraph (DAG) ──> Scheduler ──> BackendPool ──┬─> server on GPU 0
                  ▲                    │        (prefix-aware  ├─> server on GPU 1
                  └── then: spawn ─────┘         routing)      ├─> server on GPU 2
                                                                └─> server on GPU 3
```

- **TaskGraph** holds tasks and their dependencies. A task's prompt can be a
  string or a function of its dependencies' outputs. Tasks can be added
  mid-run, for example when an agent spawns subtasks.
- **Scheduler** starts a ready task whenever a slot is free, up to
  `max_concurrency`. It starts each task as soon as *its own* dependencies
  finish, not when a whole stage does. Among ready tasks it picks the one with
  the longest estimated chain of work downstream (the critical path). If a
  task fails, only the tasks that depend on it are cancelled.
- **Subagents**: give a task a `then(task, spawn)` callback. When the task
  succeeds it can spawn subtasks (`spawn.task`), a merge step (`spawn.join`),
  or both at once (`spawn.fan_out`). Spawned tasks depend on the parent,
  inherit its system prompt, and get ids like `parent/child`.
- **BackendPool** spreads requests across several servers, each with its own
  concurrency limit. With `strategy="prefix"` (the default), requests that
  share a prompt prefix (by default, the same system prompt) go to a server
  that already has it cached, unless that server is more than
  `max_imbalance` requests busier than the least-loaded one. Prompt
  processing doesn't get faster with batching, so processing a long shared
  prompt once per card instead of once per request is a large saving.
- **AdaptiveLimit** finds each server's best concurrency while it runs.
  Generation is limited by memory bandwidth, so batching more requests
  shares each read of the weights across more tokens. That helps until
  compute saturates or the KV cache fills, and where that happens has to be
  measured. The controller doubles the limit while measured tokens/s keeps
  up, then hill-climbs around the best level. Where throughput plateaus it
  picks the lowest level on the plateau. It can warm-start from a previous
  run's `curve`, which skips the costly probing phase.
- **Prompt budget**: `Endpoint(max_prefill_tokens=...)` caps uncached prompt
  tokens in flight per server. Prompt processing doesn't speed up with
  batching, so this stops a burst of long prompts stalling generation.
- **Backends** expose `async complete(request)`:
  - `OpenAICompatibleBackend` calls `/chat/completions` on any compatible server.
  - `SimulatedBackend` is a fake continuous-batching server for development
    without GPUs. Per-request speed drops as more requests share the batch,
    following a configurable speed curve. It has a small prefix cache, and
    several simulated servers can share one `SimClock`.

## Quick start

```bash
pip install -e .[dev]
python -m pytest
python examples/demo_simulated.py      # no GPU needed
python examples/demo_multi_gpu.py      # 4 simulated cards, routing strategies compared
python examples/demo_adaptive.py       # adaptive concurrency vs fixed limits
python examples/demo_server.py --base-url http://localhost:8000/v1 --model <name>
```

Four servers, one per GPU (for example four vLLM instances on ports 8000-8003):

```python
from orchestrator import AdaptiveLimit, BackendPool, Endpoint, OpenAICompatibleBackend, Scheduler

pool = BackendPool([
    Endpoint(f"gpu{i}", OpenAICompatibleBackend(f"http://localhost:{8000 + i}/v1", "qwen"),
             controller=AdaptiveLimit(max_limit=64), max_prefill_tokens=8000)
    for i in range(4)
])
report = await Scheduler(pool, max_concurrency=pool.capacity).run(graph)
```

A single simulated server:

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

### Routing across 4 cards (`examples/demo_multi_gpu.py`)

8 agent teams, each with its own ~6k-token shared system prompt, each
planning then fanning out to 8 subtasks plus a merge (80 tasks). Each
simulated card prefills at 1,800 tok/s, decodes at 46 tok/s alone and about
16 tok/s each at 64 at once, and has KV room for 2 long prefixes. Those
figures come from third-party RTX 3090 reports for a different model, so
only the comparison between strategies is meaningful.

| Strategy | Makespan | Prefix cache hits |
|---|---|---|
| round robin | 57.1s | 34 / 80 |
| least loaded | 57.1s | 34 / 80 |
| **prefix-aware** | **43.4s** | **53 / 80** |

### Adaptive concurrency (`examples/demo_adaptive.py`)

800 agent-style requests (250-2,500 token prompts, 100-500 token answers) on
4 simulated cards. "Cold" means the controllers start knowing nothing; "warm"
means they start from the curve they learned in the cold run.

| Limit per card | 27B on 3080 20GB (estimate) | Peaked curve (hypothetical) |
|---|---|---|
| fixed 4 | 501.9s | 398.6s |
| fixed 8 | 286.9s | 230.3s |
| fixed 16 | **206.4s** | 157.8s |
| fixed 32 | 209.3s | **142.5s** |
| fixed 64 | 215.2s | 147.8s |
| adaptive, cold | 252.4s | 193.5s |
| adaptive, warm | 213.6s | 147.2s |

Once warm, the controller lands within about 3% of the best fixed limit
without being told where it is. Cold probing costs 20-35% on a job this
short, which is why step 4 saves learned curves between runs. The 27B preset
(`orchestrator.presets`) is scaled from third-party RTX 3090 reports, with
an assumed ~30k-token KV cache per card. In that estimate the KV cache, not
compute, caps useful concurrency at around 16 per card.
