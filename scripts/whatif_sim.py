"""Step-level what-if simulator for the agent-team benchmark (no GPUs needed).

This is a separate, more detailed model than ``SimulatedBackend``. It
simulates vLLM at the level of scheduler steps, so it can represent the
things the 4x RTX 3090 run showed actually matter:

* chunked prefill and decode sharing each GPU step (``max_num_batched_tokens``),
* a block-based KV cache with the hybrid model's 1,568-token blocks, FCFS
  admission that stops at the first request that doesn't fit, and a
  ``max_num_seqs`` cap,
* prefix caching only in whole blocks, kept in free blocks and evicted LRU,
* tensor parallelism as per-step all-reduce cost (latency + bandwidth),
* optional LMCache-style CPU prefix cache, FP8 KV, speculative decoding,
  vLLM priority scheduling and several client-side routing policies.

The per-step cost model is

    step = eff * (t_w + n_decode * t_dec + n_prefill * t_pre + c_ctx * ctx) / tp
           + comm_scale * (n_allreduce * L + tokens * bytes_per_token / BW)

with the single-card numbers taken from published RTX 3090 measurements of
this model (46 tok/s single stream, ~300 tok/s at 8, ~1,035 tok/s at 64,
~1,800 tok/s prefill), and ``eff`` / ``comm_scale`` fitted to the measured
Vast.ai runs. See ``docs/benchmarks/whatif-sim.md`` for the calibration and
results. All numbers it prints are model estimates, not measurements.

    python scripts/whatif_sim.py calibrate
    python scripts/whatif_sim.py report     # every scenario in the write-up
"""

from __future__ import annotations

import argparse
import heapq
import math
import random
import statistics
import sys
from dataclasses import dataclass, field, replace

# --------------------------------------------------------------------------
# Workload: the same agent teams as scripts/bench.py, in tokens.

CHARS_PER_TOKEN = 4.4  # English prose with a Qwen tokenizer
BRIEF_PARAGRAPH_CHARS = 343
BRIEF_HEADER_CHARS = 70


@dataclass
class Task:
    tid: str
    team: int
    kind: str
    brief: int  # tokens of the shared team brief (system prompt)
    prompt: int  # total prompt tokens, brief included
    out: int  # output tokens (the real runs hit max_tokens almost always)
    cp: int = 0  # critical-path priority: output tokens on longest chain from here
    deps_left: int = 0
    children: list = field(default_factory=list)
    ready_at: float = 0.0
    start_at: float = 0.0
    done_at: float = 0.0
    queue_s: float = 0.0
    cached: int = 0


def build_teams(teams: int, seed: int = 0, out_jitter: float = 0.0, rng_jitter: random.Random | None = None):
    """Mirror bench.build_graph's random draws so team shapes match the real runs."""
    rng = random.Random(seed)
    out = []
    # bench.py draws each team's part count and brief length up front, and the
    # part lengths later, when each plan finishes; draw in that order.
    shapes = [(rng.randint(4, 10), rng.randint(12, 40)) for _ in range(teams)]
    part_outs = [[rng.randint(250, 600) for _ in range(n)] for n, _ in shapes]
    for t in range(teams):
        n_parts, reps = shapes[t]
        brief = int((BRIEF_HEADER_CHARS + BRIEF_PARAGRAPH_CHARS * reps) / CHARS_PER_TOKEN)
        part_out = part_outs[t]

        def j(x):
            if not out_jitter or rng_jitter is None:
                return x
            return max(16, int(x * (1 - out_jitter * rng_jitter.random())))

        plan = Task(f"t{t}/plan", t, "plan", brief, brief + 30, j(200))
        parts = [Task(f"t{t}/p{i}", t, "part", brief, brief + 40, j(m)) for i, m in enumerate(part_out)]
        review = Task(f"t{t}/review", t, "review", brief, brief + 20 + sum(p.out for p in parts), j(500))
        final = Task(f"t{t}/final", t, "final", brief, brief + 20 + review.out, j(400))
        plan.children = parts
        for p in parts:
            p.deps_left = 1
            p.children = [review]
        review.deps_left = n_parts
        review.children = [final]
        final.deps_left = 1
        final.cp = 400
        review.cp = 500 + 400
        for p in parts:
            p.cp = p.out + review.cp
        plan.cp = 200 + max(p.out for p in parts) + review.cp
        out.append([plan, *parts, review, final])
    return out


# --------------------------------------------------------------------------
# Hardware / server configuration

# One RTX 3090 running this model, before the fitted efficiency factor.
T_W = 0.0210  # s per step: weights + fixed overhead (46 tok/s single stream)
T_DEC = 0.00065  # s per decoded token in the batch (300 tok/s @8, 1035 @64)
T_PRE = 0.00055  # s per prefill token (~1,800 tok/s)
C_CTX = 0.08e-6  # s per context token read per decode step
KV_BYTES_PER_TOKEN = 4.28 * 2**30 / 60_757  # measured on the 3090s (~75 KB)
GIB = 2**30
N_ALLREDUCE = 128  # two per layer, ~64 layers
AR_BYTES_PER_TOKEN = 5120 * 2  # hidden size x bf16


@dataclass(frozen=True)
class Card:
    name: str
    vram_gib: float
    bw_scale: float  # memory bandwidth relative to a 3090
    compute_scale: float  # tensor throughput relative to a 3090


RTX3090 = Card("RTX 3090 24GB", 23.56, 1.0, 1.0)
RTX3080_20 = Card("RTX 3080 20GB", 19.5, 760 / 936, 0.84)

WEIGHTS_GIB = 16.5
CTX_GIB = 0.3  # CUDA context
ACT_GIB = 0.6  # activations / CUDA graphs, per card
GPU_UTIL = 0.92


@dataclass(frozen=True)
class Link:
    name: str
    latency_s: float  # per all-reduce
    bw_bytes: float  # effective all-reduce bus bandwidth


PCIE_NO_P2P = Link("PCIe, no P2P (via host)", 50e-6, 8e9)
PCIE_P2P = Link("PCIe 4.0 x16 with P2P", 15e-6, 20e9)


@dataclass(frozen=True)
class ServerCfg:
    card: Card = RTX3090
    tp: int = 1
    link: Link = PCIE_NO_P2P
    max_num_seqs: int = 40
    batch_tokens: int = 2048
    block: int = 1568
    kv_tokens: int | None = None  # override computed KV capacity
    kv_scale: float = 1.0  # e.g. 2.0 for an FP8 KV cache
    prefix_cache: bool = True
    cache_keep_idle: bool = True  # False: a prefix stays cached only while a request is using it
    lmcache: bool = False  # CPU prefix cache shared by all servers on the host
    lmcache_chunk: int = 256
    lmcache_bw: float = 12e9
    spec: str = "off"  # off | on | adaptive
    spec_tau: float = 2.4  # tokens per verify step (MTP, 3 drafts)
    spec_k: int = 3
    spec_t_v: float = 0.00134  # s per verified token; puts the crossover at ~8 streams
    spec_adaptive_max: int = 8
    alloc: str = "incremental"  # incremental (vLLM: grow + preempt) | reserve (prompt+max_tokens up front)
    priority: bool = False  # vLLM --scheduling-policy priority, critical path first
    eff: float = 1.0
    comm_scale: float = 1.0

    def kv_capacity(self) -> int:
        if self.kv_tokens is not None:
            return int(self.kv_tokens * self.kv_scale)
        per_card = self.card.vram_gib * GPU_UTIL - WEIGHTS_GIB / self.tp - CTX_GIB - ACT_GIB
        per_card = max(per_card, 0.05)
        return int(per_card * self.tp / (KV_BYTES_PER_TOKEN / GIB) * self.kv_scale)


@dataclass
class Req:
    task: Task
    prefill_left: float
    decode_left: float
    blocks: int  # private blocks held
    shared: int = 0  # blocks shared through a prefix-cache entry
    registered: bool = False
    load_s: float = 0.0
    ctx: float = 0.0  # tokens in the sequence so far (prompt + generated)
    generated: float = 0.0
    prio: tuple = ()


class Server:
    def __init__(self, cfg: ServerCfg, name: str, cpu_cache: set | None, rng: random.Random, jitter: float):
        self.cfg = cfg
        self.name = name
        self.total_blocks = max(1, cfg.kv_capacity() // cfg.block)
        self.private = 0
        self.cache: dict[int, list] = {}  # team -> [blocks, refcount, last_used]
        self.waiting: list = []  # heap of (prio, Req-or-Task)
        self.running: list[Req] = []
        self.seq = 0
        self.busy_until: float | None = None
        self.cpu_cache = cpu_cache
        self.rng = rng
        self.jitter = jitter
        self.peak_running = 0
        self.preemptions = 0
        self.wasted_tokens = 0.0
        self.outstanding_tokens = 0
        self.outstanding = 0
        self.busy_s = 0.0
        c = cfg.card
        k = cfg.tp
        self.t_w = T_W / c.bw_scale / k * cfg.eff
        self.t_dec = T_DEC / c.compute_scale / k * cfg.eff
        self.t_pre = T_PRE / c.compute_scale / k * cfg.eff
        self.t_v = cfg.spec_t_v / c.compute_scale / k * cfg.eff
        self.c_ctx = C_CTX / c.bw_scale / k * cfg.eff
        if k > 1:
            ring = 2 * (k - 1) / k
            self.comm_fixed = cfg.comm_scale * N_ALLREDUCE * cfg.link.latency_s
            self.comm_tok = cfg.comm_scale * N_ALLREDUCE * AR_BYTES_PER_TOKEN * ring / cfg.link.bw_bytes
        else:
            self.comm_fixed = self.comm_tok = 0.0

    # -- KV blocks and prefix cache --
    def free_blocks(self) -> int:
        return self.total_blocks - self.private - sum(e[0] for e in self.cache.values())

    def _evict_for(self, n: int) -> bool:
        while self.free_blocks() < n:
            idle = [(e[2], team) for team, e in self.cache.items() if e[1] == 0]
            if not idle:
                return False
            _, team = min(idle)
            del self.cache[team]
        return True

    def enqueue(self, task: Task, now: float) -> None:
        self.seq += 1
        prio = (-task.cp if self.cfg.priority else 0, self.seq)
        r = Req(task, 0, float(task.out), 0, prio=prio)
        heapq.heappush(self.waiting, (prio, self.seq, r))
        self.outstanding += 1
        self.outstanding_tokens += task.prompt + task.out

    def _release(self, r: Req, now: float) -> None:
        self.private -= r.blocks
        r.blocks = 0
        if r.shared:
            e = self.cache.get(r.task.team)
            if e is not None:
                e[1] -= 1
                e[2] = now
                if e[1] <= 0 and not self.cfg.cache_keep_idle:
                    del self.cache[r.task.team]
            r.shared = 0

    def _preempt(self, r: Req, now: float) -> None:
        """vLLM-style preemption by recompute: drop the KV, requeue at the front."""
        self.running.remove(r)
        self._release(r, now)
        self.preemptions += 1
        self.wasted_tokens += r.ctx - r.task.cached
        r.registered = False
        # Front of its priority class: vLLM puts preempted requests back first.
        heapq.heappush(self.waiting, ((r.prio[0], -1e9 + self.preemptions), self.seq, r))

    def _try_admit(self, r: Req, now: float) -> bool:
        cfg = self.cfg
        task = r.task
        if len(self.running) >= cfg.max_num_seqs:
            return False
        seq_len = task.prompt + int(r.generated)
        if cfg.alloc == "reserve":
            need = math.ceil((task.prompt + task.out) / cfg.block)
        else:
            need = math.ceil((seq_len + 1) / cfg.block)
        cached = 0
        shared = 0
        entry = self.cache.get(task.team) if cfg.prefix_cache else None
        if entry is not None:
            shared = entry[0]
            cached = shared * cfg.block
        private = max(0, need - shared)
        if private > self.total_blocks and not self.running:
            private = self.total_blocks  # always admit into an empty server
        if not self._evict_for(private):
            return False
        load_s = 0.0
        if cfg.lmcache and self.cpu_cache is not None and task.team in self.cpu_cache:
            lm = (task.brief // cfg.lmcache_chunk) * cfg.lmcache_chunk
            if lm > cached:
                load_s = (lm - cached) * KV_BYTES_PER_TOKEN / cfg.lmcache_bw
                cached = lm
        if entry is not None:
            entry[1] += 1
            entry[2] = now
        self.private += private
        r.blocks = private
        r.shared = shared
        r.load_s = load_s
        r.prefill_left = seq_len - cached
        r.ctx = cached
        if r.generated == 0:
            task.cached = cached
            task.start_at = now
            task.queue_s = now - task.ready_at
        return True

    def has_work(self) -> bool:
        return bool(self.running or self.waiting)

    def _victim(self) -> Req:
        # Lowest priority, latest admitted.
        return max(self.running, key=lambda q: (q.prio[0], self.running.index(q)))

    def step(self, now: float) -> float:
        """Schedule one step starting at `now`; return its duration. Progress applies at the end."""
        cfg = self.cfg
        n_dec_guess = sum(1 for r in self.running if r.prefill_left <= 0)
        use_spec = cfg.spec == "on" or (cfg.spec == "adaptive" and n_dec_guess <= cfg.spec_adaptive_max)
        per = cfg.spec_tau if use_spec else 1.0
        # Grow decoders' KV by a block when they cross a boundary; preempt if full.
        if cfg.alloc == "incremental":
            for r in list(self.running):
                if r not in self.running or r.prefill_left > 0:
                    continue
                if r.ctx + per > (r.blocks + r.shared) * cfg.block:
                    while not self._evict_for(1):
                        v = self._victim()
                        self._preempt(v, now)
                        if v is r:
                            break
                    if r in self.running:
                        self.private += 1
                        r.blocks += 1
        decoders = [r for r in self.running if r.prefill_left <= 0]
        n_dec = len(decoders)
        budget = cfg.batch_tokens
        dec_cost_tokens = n_dec * (cfg.spec_k + 1 if use_spec else 1)
        budget -= dec_cost_tokens
        plan = []
        load_s = 0.0
        for r in self.running:
            if r.prefill_left > 0 and budget > 0:
                c = min(budget, r.prefill_left)
                plan.append((r, c))
                budget -= c
        while self.waiting and budget > 0:
            _, _, r = self.waiting[0]
            if not self._try_admit(r, now):
                break  # the head blocks the queue, as in vLLM
            heapq.heappop(self.waiting)
            self.running.append(r)
            load_s += r.load_s
            c = min(budget, r.prefill_left)
            if c > 0:
                plan.append((r, c))
                budget -= c
        self.peak_running = max(self.peak_running, len(self.running))
        n_pre = sum(c for _, c in plan)
        ctx = sum(r.ctx for r in decoders)
        if use_spec:
            dec_t = n_dec * (cfg.spec_k + 1) * self.t_v
        else:
            dec_t = n_dec * self.t_dec
        dt = self.t_w + dec_t + n_pre * self.t_pre + ctx * self.c_ctx
        dt += self.comm_fixed + self.comm_tok * (dec_cost_tokens + n_pre) + load_s
        if self.jitter:
            dt *= math.exp(self.rng.gauss(0, self.jitter))
        self.busy_s += dt
        self._pending = (plan, decoders, per)
        return dt

    def finish_step(self, now: float) -> list[Task]:
        plan, decoders, per = self._pending
        cfg = self.cfg
        done = []
        for r in decoders:
            if r in self.running:
                r.decode_left -= per
                r.generated += per
                r.ctx += per
        for r, c in plan:
            if r not in self.running:
                continue
            r.prefill_left -= c
            r.ctx += c
            if r.prefill_left <= 0:
                r.decode_left -= 1  # the prefill step emits the first token
                r.generated += 1
                r.ctx += 1
                t = r.task
                if cfg.lmcache and self.cpu_cache is not None:
                    self.cpu_cache.add(t.team)
                if cfg.prefix_cache and not r.registered and not r.shared and t.team not in self.cache:
                    nb = min(t.brief // cfg.block, r.blocks)
                    if nb > 0:
                        # This request's brief blocks become a shared cache entry.
                        self.private -= nb
                        r.blocks -= nb
                        r.shared = nb
                        self.cache[t.team] = [nb, 1, now]
                        r.registered = True
        for r in list(self.running):
            if r.prefill_left <= 0 and r.decode_left <= 0:
                self.running.remove(r)
                self._release(r, now)
                r.task.done_at = now
                self.outstanding -= 1
                self.outstanding_tokens -= r.task.prompt + r.task.out
                done.append(r.task)
        return done


# --------------------------------------------------------------------------
# Client side: dependency scheduler + routing


@dataclass(frozen=True)
class Scenario:
    label: str
    servers: tuple[ServerCfg, ...]
    routing: str = "rr"  # rr | lor | lot | affinity | prefix
    client_limit: int | None = None  # per-server cap held in the orchestrator (managed mode)
    prefill_budget: int | None = None
    team_order: bool = False  # client priority: earlier teams first (in managed mode)


def simulate(sc: Scenario, teams: int = 24, seed: int = 0, run_seed: int = 0, jitter: float = 0.0,
             out_jitter: float = 0.0) -> dict:
    rng = random.Random(run_seed)
    team_tasks = build_teams(teams, seed, out_jitter, random.Random(run_seed + 1000))
    cpu_cache: set = set()
    servers = [Server(cfg, f"s{i}", cpu_cache, random.Random(run_seed * 31 + i), jitter)
               for i, cfg in enumerate(sc.servers)]
    n = len(servers)
    rr = [0]
    affinity: dict[int, int] = {}
    warm: dict[int, set] = {i: set() for i in range(n)}
    events: list = []  # (time, seq, server index)
    evseq = [0]
    now = 0.0
    all_tasks = [t for tt in team_tasks for t in tt]
    client_q: list = []  # managed mode heap
    in_flight = [0] * n

    def pick(task: Task) -> int | None:
        cands = list(range(n))
        if sc.client_limit is not None:
            cands = [i for i in cands if in_flight[i] < sc.client_limit]
            if sc.prefill_budget is not None:
                def fits(i):
                    if in_flight[i] == 0:
                        return True
                    unc = task.prompt - (task.brief if task.team in warm[i] else 0)
                    return prefill_in_flight[i] + unc <= sc.prefill_budget
                cands = [i for i in cands if fits(i)]
            if not cands:
                return None
        if sc.routing == "rr":
            for _ in range(n):
                i = rr[0] % n
                rr[0] += 1
                if i in cands:
                    return i
            return cands[0]
        if sc.routing == "lor":
            return min(cands, key=lambda i: (servers[i].outstanding, i))
        if sc.routing == "lot":
            return min(cands, key=lambda i: (servers[i].outstanding_tokens, i))
        if sc.routing == "affinity":
            if task.team not in affinity:
                affinity[task.team] = min(range(n), key=lambda i: (servers[i].outstanding_tokens, i))
            i = affinity[task.team]
            return i if i in cands else None
        if sc.routing == "prefix":
            least = min(cands, key=lambda i: (in_flight[i], i))
            w = [i for i in cands if task.team in warm[i]]
            if w:
                best = min(w, key=lambda i: in_flight[i])
                if in_flight[best] - in_flight[least] <= 4:
                    return best
            return least
        raise ValueError(sc.routing)

    managed_running: dict[int, list] = {i: [] for i in range(n)}
    prefill_in_flight = [0] * n
    unc_of: dict[str, int] = {}

    def kick(i: int) -> None:
        s = servers[i]
        if s.busy_until is None and s.has_work():
            dt = s.step(now)
            s.busy_until = now + dt
            evseq[0] += 1
            heapq.heappush(events, (s.busy_until, evseq[0], i))

    def send(task: Task, i: int) -> None:
        unc_of[task.tid] = task.prompt - (task.brief if task.team in warm[i] else 0)
        prefill_in_flight[i] += unc_of[task.tid]
        warm[i].add(task.team)
        in_flight[i] += 1
        managed_running[i].append(task)
        servers[i].enqueue(task, now)
        kick(i)

    cseq = [0]

    def ready(task: Task) -> None:
        task.ready_at = now
        if sc.client_limit is None:
            send(task, pick(task))
        else:
            cseq[0] += 1
            key = (task.team if sc.team_order else 0, -task.cp, cseq[0])
            heapq.heappush(client_q, (key, task))

    def drain_client() -> None:
        if sc.client_limit is None:
            return
        held = []
        while client_q:
            key, task = heapq.heappop(client_q)
            i = pick(task)
            if i is None:
                held.append((key, task))
                if sc.routing != "affinity":
                    break  # strict priority, as the orchestrator does
                continue
            send(task, i)
        for h in held:
            heapq.heappush(client_q, h)

    for tt in team_tasks:
        ready(tt[0])
    drain_client()
    for i in range(n):
        kick(i)

    while events:
        now, _, i = heapq.heappop(events)
        s = servers[i]
        s.busy_until = None
        for t in s.finish_step(now):
            in_flight[i] -= 1
            prefill_in_flight[i] -= unc_of[t.tid]
            managed_running[i].remove(t)
            for c in t.children:
                c.deps_left -= 1
                if c.deps_left == 0:
                    ready(c)
        drain_client()
        kick(i)

    makespan = max(t.done_at for t in all_tasks)
    prompt = sum(t.prompt for t in all_tasks)
    cached = sum(t.cached for t in all_tasks)
    outs = sum(t.out for t in all_tasks)
    return {
        "label": sc.label,
        "makespan": makespan,
        "tok_s": outs / makespan,
        "cache_pct": 100 * cached / prompt,
        "peak_running": [s.peak_running for s in servers],
        "kv_blocks": servers[0].total_blocks,
        "preemptions": sum(s.preemptions for s in servers),
        "wasted_pct": 100 * sum(s.wasted_tokens for s in servers) / prompt,
        "util": statistics.mean(s.busy_s for s in servers) / makespan,
        "queue_s": statistics.mean(t.queue_s for t in all_tasks),
        "tasks": len(all_tasks),
        "avg_prompt": prompt / len(all_tasks),
        "avg_out": outs / len(all_tasks),
    }


def replicas(n=4, **kw) -> tuple[ServerCfg, ...]:
    return tuple(ServerCfg(**kw) for _ in range(n))


def run_seeds(sc: Scenario, teams: int, seeds=range(5), jitter=0.04, out_jitter=0.0) -> tuple[float, float, dict]:
    res = [simulate(sc, teams, run_seed=s, jitter=jitter, out_jitter=out_jitter) for s in seeds]
    ms = [r["makespan"] for r in res]
    return statistics.mean(ms), (statistics.stdev(ms) if len(ms) > 1 else 0.0), res[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "report"])
    a = ap.parse_args()
    import whatif_report  # noqa: F401  (scenarios live next to this file)

    whatif_report.main(a.cmd)
