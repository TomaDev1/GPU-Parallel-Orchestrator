# GPU Parallel Orchestrator

**Run multi-agent LLM workloads as parallel as their dependencies allow, across one or more local inference servers.**

You describe a job as a graph of tasks: plan, fan out into subtasks, review, summarise. The orchestrator starts every task the moment its own inputs are ready and spreads the work across your GPU servers. It keeps enough requests in flight that a batching server (vLLM, SGLang, llama.cpp, or any OpenAI-compatible endpoint) can batch them. The goal is the shortest time to finish the whole job, not the most agents running at once.

<p align="center">
  <img src="docs/benchmarks/vast-4x3090-makespan.svg" alt="One model copy per GPU, driven by the orchestrator, finished an agent workload 1.4x faster than stock vLLM split across 4 GPUs" width="760">
</p>

<p align="center"><sub>Measured on 4× RTX 3090 with Qwen3.8-27B (4-bit). Full results in <a href="#benchmark-results">Benchmark results</a>.</sub></p>

## Key results

> **1.4× faster than stock vLLM** on the same 4 GPUs: 177 s vs 257.6 s for a 239-task agent workload, by running one model copy per GPU instead of one server split across all four. ([details](#benchmark-results))

> **Memory, not compute, set the ceiling.** The KV cache capped each card at ~20 concurrent requests, so the orchestrator's own concurrency limits added nothing over vLLM's batching.

## When to use it

**It beats plain vLLM when:**

- you have **several GPUs without NVLink** and the model fits on one card (or on a pair) with room to spare: run a copy per card or pair and let the orchestrator pool them (1.4× in our test);
- your work is **multi-step agent work with dependencies**, where tasks should start the moment their inputs are ready instead of stage by stage;
- you want to **see where the time goes**: timelines, the critical path and saved metrics for every run.

**It won't beat vLLM at:**

- **batching on a single server.** vLLM's own scheduler batched as well as any limit we set, so leave the orchestrator's limits at the server's `--max-num-seqs`;
- **jobs bound by a dependency chain.** The last ~30% of every run was reviews and summaries waiting on each other. More GPUs or bigger batches can't shorten that; faster single-stream generation can.

**Prefix-aware routing is unproven on real hardware so far.** It helped a lot in simulation, but only pays off when the server has KV room to keep shared prompts cached. That wasn't the case with this hybrid model on 24 GB cards. Expect it to matter more with non-hybrid models or cards with more memory.

### Choosing a layout

| Your situation | Try first |
| --- | --- |
| The model leaves several GB of KV room on one card | One copy per GPU |
| The model barely fits on one card (e.g. a 27B 4-bit model on 20 GB cards) | Two servers of 2 GPUs each (TP2), then compare |
| The model doesn't fit on two cards, or you have NVLink | One tensor-parallel server |

Whichever you pick, measure it: [`scripts/bench.py`](#benchmarking-your-own-setup) runs the same workload against any of them.

## Highlights

- **Dependency-driven scheduling.** A task starts as soon as *its own* dependencies finish, never when a whole stage does. Among ready tasks, the one with the longest chain of work after it goes first.
- **Agents that spawn agents.** A finished task can add subtasks, a merge step, or a whole fan-out while the job is running.
- **Many servers, one pool.** Run one server per GPU (or per GPU pair) and the pool routes each request to a server, preferring the one that already has its prompt prefix cached.
- **Measured, not guessed.** An optional controller learns each server's best concurrency from live throughput. Every run can be saved to SQLite and drawn as an HTML timeline with the critical path highlighted.
- **Develop without GPUs.** A simulated continuous-batching server with a shared clock runs whole workloads in seconds.
- **No required dependencies.** Pure Python 3.10+ and asyncio; `httpx` only for real HTTP servers.

## Quick start

```bash
git clone https://github.com/TomaDev1/GPU-Parallel-Orchestrator
cd GPU-Parallel-Orchestrator
pip install -e .[dev]
python -m pytest
```

Try it with no GPU:

```bash
python examples/demo_simulated.py      # one simulated server
python examples/demo_multi_gpu.py      # 4 simulated cards, routing strategies compared
python examples/demo_adaptive.py       # adaptive concurrency vs fixed limits
python examples/demo_metrics.py        # two runs with a metrics DB, then timeline.html
```

Against a real server:

```bash
python examples/demo_server.py --base-url http://localhost:8000/v1 --model <name>
```

## Usage

### A task graph

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

A prompt can be a string or a function of the dependencies' outputs. `cpu` and `gpu` run in parallel the moment `research` finishes.

### Subagents

```python
def plan_then(task, spawn):
    parts = [spawn.task(f"part{i}", f"Write section {i + 1}.") for i in range(6)]
    spawn.join("review", lambda d: "Review:\n\n" + "\n\n".join(d.values()), of=parts)

g.add(Task("plan", "Plan a 6-section report.", system=BRIEF, then=plan_then))
```

Spawned tasks depend on their parent, inherit its system prompt (so they share its cached prefix), and get ids like `plan/part3`.

### Several servers

One OpenAI-compatible server per GPU, for example four vLLM instances on ports 8000–8003:

```python
from orchestrator import BackendPool, Endpoint, OpenAICompatibleBackend, Scheduler

pool = BackendPool([
    Endpoint(f"gpu{i}", OpenAICompatibleBackend(f"http://localhost:{8000 + i}/v1", "qwen"),
             max_concurrency=40)          # match the server's own --max-num-seqs
    for i in range(4)
])
report = await Scheduler(pool, max_concurrency=pool.capacity).run(graph)
```

> [!TIP]
> On vLLM, set each endpoint's `max_concurrency` to the server's `--max-num-seqs` and let vLLM do the batching. In our tests, lower limits only added queueing (see [Benchmark results](#benchmark-results)).

### Metrics, learned limits and timelines

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

`store.controller(name)` warm-starts from the speed curve learned in earlier runs, so only the first run pays for probing. Keep endpoint names stable, since saved curves are keyed by name. Start a new database after changing the model or quantisation.

## How it works

```
objective ──> TaskGraph (DAG) ──> Scheduler ──> BackendPool ──┬─> server on GPU 0
                  ▲                    │        (prefix-aware  ├─> server on GPU 1
                  └── then: spawn ─────┘         routing)      ├─> server on GPU 2
                                                                └─> server on GPU 3
```

| Component | What it does |
| --- | --- |
| `TaskGraph` | Holds tasks and dependencies. Tasks can be added mid-run, for example when an agent spawns subtasks. |
| `Scheduler` | Starts a ready task whenever a slot is free, choosing the one with the longest estimated chain of work after it. A failure cancels only the tasks that depend on it. |
| `Spawner` (`then=`) | `spawn.task`, `spawn.join` and `spawn.fan_out` add subtasks and merge steps from a finished task. |
| `BackendPool` | Spreads requests over several servers, each with its own concurrency limit. With `strategy="prefix"` (the default), requests sharing a prompt prefix go to a server that already has it cached, unless that server is more than `max_imbalance` requests busier than the least-loaded one. |
| `AdaptiveLimit` | Optional. Doubles a server's limit while measured throughput keeps up, then hill-climbs around the best level and picks the lowest level on a plateau. |
| `Endpoint(max_prefill_tokens=…)` | Optional cap on uncached prompt tokens in flight per server, so a burst of long prompts can't stall generation. |
| `MetricsStore` | SQLite record of every run: per-task timings, which server ran each task, controller history, learned speed curves. |
| `report.critical_path()` | The chain of tasks that decided the total time, traced back from the last task to finish. |
| `write_timeline()` | A single self-contained HTML file showing each server's tasks over time, their queue waits, and the critical path. |
| `OpenAICompatibleBackend` | Calls `/chat/completions` on any compatible server and reads prefix-cache hits from vLLM's usage details. |
| `SimulatedBackend` | A fake continuous-batching server for development: per-request speed falls as the batch grows, with a prefix cache, a KV cache limit and optional batch slots. |

## Benchmarking your own setup

`scripts/launch_vllm.sh` starts one vLLM server per GPU, and `scripts/bench.py` runs a fixed agent workload against them. Every run uses the same workload and seed, so results compare directly.

```bash
# One vLLM server per GPU on ports 8000+ (logs in logs/, stop with: ./scripts/launch_vllm.sh stop)
MODEL=RedHatAI/Qwen3.8-27B-INT4 MAX_MODEL_LEN=8192 \
EXTRA_ARGS='--max-num-seqs 40 --max-num-batched-tokens 2048' \
./scripts/launch_vllm.sh

# Fixed limits, then adaptive runs, then a run frozen at the learned limit
python scripts/bench.py --limits 8,16,32

# Baseline: every ready task sent straight to the servers, no limits or routing
python scripts/bench.py --unmanaged --limits "" --adaptive-runs 0

# A single server spanning several GPUs (e.g. vLLM --tensor-parallel-size 4)
python scripts/bench.py --endpoints http://localhost:8000/v1 --unmanaged --limits "" --adaptive-runs 0
```

Results land in `bench-results/` (`summary.json` and one `timeline-runN.html` per run) and `bench.db`. Add `--teams 96` for a heavier workload. `python scripts/bench.py --simulate` shows what the output looks like without GPUs.

<details>
<summary><b>Tips for vLLM and rented GPUs</b></summary>

- Use a 4-bit build made for vLLM (AWQ, GPTQ, W4A16 or compressed-tensors). GGUF repos don't load in vLLM.
- Leave at least 60 GB of disk for the model download.
- Hybrid models such as Qwen3.8 need `--max-num-seqs` at or below the number of Mamba cache blocks vLLM reports at startup, or the server refuses to start.
- If a server runs out of memory while profiling, add `--max-num-batched-tokens 2048` and set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- The launcher refuses to start if a GPU is already busy (for example a template's own vLLM). Stop that server first, or set `FORCE=1`.
- Work inside `tmux` so a dropped SSH connection doesn't stop a run.

</details>

## Benchmark results

> Measured on a rented machine with 4× RTX 3090 24GB (no NVLink), running [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) on vLLM 0.32.0. The workload is 24 agent teams (239 tasks) that plan, fan out into 4–10 writers, then review and summarise. The full write-up, every run and the raw data are in [docs/benchmarks](docs/benchmarks/vast-4x3090.md).

### One copy per GPU beat one big server by 1.4×

| Setup | Time to finish | Speed-up |
| --- | ---: | ---: |
| Stock vLLM, one server across 4 GPUs (`--tensor-parallel-size 4`) | 257.6 s | 1.00× |
| 4 copies + orchestrator, limit learned automatically (8 per GPU) | 183.0 s | 1.41× |
| 4 copies + orchestrator, no limits | **177.1 s** | **1.45×** |

<p align="center">
  <img src="docs/benchmarks/layouts.svg" alt="Three layouts for one model on 4 GPUs: one copy per GPU, two tensor-parallel pairs, and one tensor-parallel server across all four" width="760">
</p>

Without NVLink, a tensor-parallel server synchronises every card over PCIe at every step. Four independent copies skip that entirely, and the orchestrator is what turns four separate servers into one pool: it keeps all of them busy and starts each agent task the moment its inputs exist.

### The limits didn't matter; memory did

Every per-GPU limit from 8 up finished within noise (177–195 s), and the coordinator's prefix-aware routing didn't beat round-robin at 96 teams either (666 s vs 653 s). The vLLM logs show why: each card's KV cache filled at about **20 running requests**, half the 40 allowed. vLLM queued the rest itself, and prefix-cache hits stayed at 7–13% because cached prompts were evicted almost as soon as they were stored.

<p align="center">
  <img src="docs/benchmarks/vast-4x3090-kv-limit.svg" alt="Requests running on one GPU stayed at about 20 throughout, with the rest waiting inside vLLM" width="760">
</p>

With 16.8 GB of weights on a 24 GB card, only 4.3 GB was left for the KV cache. The model is a hybrid (attention plus Mamba-style layers), so vLLM caches in 1,568-token blocks, and each request needed about two of the ~38 blocks available.

<details>
<summary><b>Simulator results (illustrative only)</b></summary>

These came from `SimulatedBackend` with speed curves scaled from third-party reports, before any real hardware was tested. They show how the pieces behave; they are not predictions. The simulator assumed prefix caching always works, and on real hardware it didn't (see above).

**Routing across 4 cards** (`examples/demo_multi_gpu.py`, 8 agent teams sharing ~6k-token prompts, 80 tasks):

| Strategy | Makespan | Prefix cache hits |
|---|---|---|
| round robin | 57.1s | 34 / 80 |
| least loaded | 57.1s | 34 / 80 |
| **prefix-aware** | **43.4s** | **53 / 80** |

**Adaptive concurrency** (`examples/demo_adaptive.py`, 800 requests on 4 simulated cards):

| Limit per card | Preset A | Preset B (peaked curve) |
|---|---|---|
| fixed 4 | 501.9s | 398.6s |
| fixed 8 | 286.9s | 230.3s |
| fixed 16 | **206.4s** | 157.8s |
| fixed 32 | 209.3s | **142.5s** |
| fixed 64 | 215.2s | 147.8s |
| adaptive, cold | 252.4s | 193.5s |
| adaptive, warm | 213.6s | 147.2s |

Once warm, the controller lands within about 3% of the best fixed limit without being told where it is.

</details>
