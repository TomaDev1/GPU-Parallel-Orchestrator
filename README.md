# GPU-Parallel-Orchestrator

A coordinator for multi-agent LLM work that keeps a local GPU inference server
as busy as is useful. It breaks a job into a graph of tasks and runs every task
the moment its inputs are ready. It keeps several requests in flight so a
batching server (Strata, vLLM, SGLang, llama.cpp) can process them together.

The goal is the shortest total time to finish the whole job, not the most
agents running at once.

## Status

**All 4 planned steps done.** Built and tested against simulated servers. The target
setup is an EPYC workstation with 4× RTX 3080 20GB, each running its own copy
of Qwen3.8-27B at 4-bit. It isn't built yet.

| Step | What | State |
|---|---|---|
| 1 | Task graph, dependency-driven scheduler, backend interface, simulator | done |
| 2 | Multiple servers (one per GPU), prefix-aware routing, subagent spawning API | done |
| 3 | Adaptive admission controller: tune concurrency per server from measured throughput; separate prompt-token budget | done |
| 4 | SQLite metrics store, learned curves saved between runs, critical-path analysis, HTML timeline | done |

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
- **MetricsStore** saves every run to SQLite: per-task timings, which
  server ran each task, controller history, and each server's learned speed
  curve. `store.controller("gpu0")` returns a controller warm-started from
  that curve, so only the first run pays for probing.
- **Critical path**: `report.critical_path()` walks back from the last task
  to finish through whichever dependency it was really waiting on. That
  chain decided the total time, so it's where splitting work or giving
  priority pays off.
- **Timeline**: `write_timeline(report, "timeline.html")` writes a single
  HTML file. It shows each server's tasks over time, how long each waited
  for a slot, and the critical path highlighted.
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
python examples/demo_metrics.py        # two runs with a metrics DB, then timeline.html
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

With metrics and warm-started controllers:

```python
from orchestrator import MetricsStore, write_timeline

with MetricsStore("metrics.db") as store:
    pool = BackendPool([
        Endpoint(f"gpu{i}", OpenAICompatibleBackend(f"http://localhost:{8000 + i}/v1", "qwen"),
                 controller=store.controller(f"gpu{i}", max_limit=64))
        for i in range(4)
    ])
    report = await Scheduler(pool, max_concurrency=pool.capacity).run(graph)
    store.save_run(report, pool, label="research batch")
write_timeline(report, "timeline.html")
```

Keep endpoint names stable: they're the key for saved curves. Use new names
(or a new database) after changing the model or quantisation.

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

## Testing on Vast.ai

You can measure real numbers before your own machine is built by renting a
4-GPU instance. 4× RTX 3090 is the closest common match: a bit faster than a
3080 20GB, with more VRAM for KV cache, but the same shape of behaviour.

1. **Rent the instance.** Filter for 4 GPUs of the same type, pick a vLLM
   template (or any recent CUDA image and `pip install vllm`), and give it
   about 60GB of disk. Connect over SSH and work inside `tmux` so a dropped
   connection doesn't kill the run.
2. **Get the code.**
   ```bash
   git clone https://github.com/TomaDev1/GPU-Parallel-Orchestrator && cd GPU-Parallel-Orchestrator
   pip install -e .[dev]
   ```
3. **Start one vLLM server per GPU.** Use a 4-bit (AWQ or GPTQ) build of
   Qwen3.8-27B from Hugging Face. The model downloads once, then four servers
   start on ports 8000-8003. Logs are in `logs/`.
   ```bash
   MODEL=<hf-repo-of-a-4bit-qwen3.8-27b> KV_DTYPE=fp8 ./scripts/launch_vllm.sh
   ```
   If a server runs out of memory, lower `MAX_MODEL_LEN` (default 16384) or
   `GPU_MEM` (default 0.92). For a gated model, set `HF_TOKEN` first.
4. **Run the benchmark.** Every run uses the same agent workload, so results
   compare directly.
   ```bash
   python scripts/bench.py --limits 4,8,16,32
   ```
   This runs the fixed limits, then three adaptive runs (the first one probes
   and later ones start warm), then one run frozen at each GPU's best learned
   level. It prints a table and each GPU's learned speed curve. Everything is
   in `bench-results/` (`summary.json`, a `timeline-runN.html` per run) and
   `bench.db`.
5. **Bring the results home** (`scp -r` the folder, or download it through
   the Vast web UI), then stop the instance with `./scripts/launch_vllm.sh stop`
   and destroy it in the Vast console so billing stops.

Try `python scripts/bench.py --simulate --limits 8,16,32` locally first to
see what the output looks like.

What to look for: where each GPU's curve flattens (that's its real
concurrency limit), whether the frozen learned run matches the best fixed
limit, and in the timelines, how much of the critical path is waiting rather
than running. On short jobs, exploring costs more than it gains, so running
frozen at the learned level is usually the better everyday setting. The
adaptive mode earns its keep on long or changing workloads.

## When the hardware arrives

1. Start one vLLM (or llama.cpp) server per card, on ports 8000-8003, each
   pinned to its GPU with `CUDA_VISIBLE_DEVICES`.
2. Run a representative job through a `BackendPool` with `store.controller(...)`
   per card, and save it with `MetricsStore`. The first run probes; later
   runs start warm.
3. Open the timeline to see where tasks waited, and compare the learned
   curves against the simulator presets. Then replace the presets with the
   real numbers.
