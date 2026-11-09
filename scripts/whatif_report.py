"""Scenarios for the what-if write-up (docs/benchmarks/whatif-sim.md).

    python scripts/whatif_sim.py calibrate
    python scripts/whatif_sim.py report > results.txt

Every number printed is a simulator estimate calibrated to the 2026-11-08
Vast.ai 4x RTX 3090 runs, not a measurement.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import replace

import whatif_sim as w

# Fitted on run 10 (4 copies, unmanaged, 177.1 s) and run 9 (TP4, 257.6 s).
EFF = 1.1506
COMM = 1.5004
# Alternative cache model (prefixes are not kept once idle), refitted the same way.
EFF_B = 1.1146
COMM_B = 1.0472

MEASURED = {
    ("4 copies, round-robin (measured baseline)", 24): 177.1,
    ("TP4 stock, no P2P", 24): 257.6,
    ("limit 4, prefix routing", 24): 241.2,
    ("limit 8, prefix routing", 24): 192.2,
    ("limit 16, prefix routing", 24): 195.4,
    ("limit 32, prefix routing", 24): 187.2,
    ("4 copies, round-robin (measured baseline)", 96): 652.6,
    ("limit 40, prefix routing", 96): 666.2,
}

# "GPU unlock": raised power limit + memory overclock. Decode is memory-bound,
# so a ~7% memory clock gain is roughly the ceiling; prefill gains ~5% from clocks.
UNLOCKED_3090 = replace(w.RTX3090, name="RTX 3090 unlocked", bw_scale=1.07, compute_scale=1.05)
CARD_3080 = replace(w.RTX3080_20, vram_gib=19.8)  # ~0.8 GiB KV left per copy, as estimated in vast-4x3090.md
UNLOCKED_3080 = replace(CARD_3080, name="RTX 3080 20GB unlocked",
                        bw_scale=CARD_3080.bw_scale * 1.07, compute_scale=CARD_3080.compute_scale * 1.05)


def base(**kw) -> w.ServerCfg:
    d = {"eff": EFF, "comm_scale": COMM}
    d.update(kw)
    return w.ServerCfg(**d)


def rep(n=4, **kw):
    return tuple(base(**kw) for _ in range(n))


def tp(k, n, **kw):
    kw.setdefault("max_num_seqs", 256 if k == 4 else 128)
    return tuple(base(tp=k, **kw) for _ in range(n))


HQ = {"eff": 1.0}  # HyperQwen-patched kernels: the published per-card speeds, no stock-vLLM penalty
HQ_SPEC = {"eff": 1.0, "spec": "adaptive"}
HQ_FULL = {"eff": 1.0, "spec": "adaptive", "block": 256}  # + prefix caching at fine granularity


def scenarios_3090():
    S = w.Scenario
    out = [
        ("Calibration", S("4 copies, round-robin (measured baseline)", rep())),
        ("Calibration", S("TP4 stock, no P2P", tp(4, 1))),
        ("Calibration", S("limit 4, prefix routing", rep(), "prefix", 4, 16000)),
        ("Calibration", S("limit 8, prefix routing", rep(), "prefix", 8, 16000)),
        ("Calibration", S("limit 16, prefix routing", rep(), "prefix", 16, 16000)),
        ("Calibration", S("limit 32, prefix routing", rep(), "prefix", 32, 16000)),
        ("Calibration", S("limit 40, prefix routing", rep(), "prefix", 40, 16000)),
        # Layout
        ("Layout", S("TP4 + P2P", tp(4, 1, link=w.PCIE_P2P))),
        ("Layout", S("TP4 tuned (8k batch tokens), no P2P", tp(4, 1, batch_tokens=8192))),
        ("Layout", S("TP4 + P2P, tuned", tp(4, 1, link=w.PCIE_P2P, batch_tokens=8192))),
        ("Layout", S("2x TP2, no P2P, least-outstanding", tp(2, 2), "lor")),
        ("Layout", S("2x TP2 + P2P, least-outstanding", tp(2, 2, link=w.PCIE_P2P), "lor")),
        ("Layout", S("2x TP2 + P2P, team affinity", tp(2, 2, link=w.PCIE_P2P), "affinity")),
        # Routing (4 copies, no client limits)
        ("Routing", S("4 copies, least-outstanding requests", rep(), "lor")),
        ("Routing", S("4 copies, least-outstanding tokens", rep(), "lot")),
        ("Routing", S("4 copies, team affinity", rep(), "affinity")),
        # Scheduling
        ("Scheduling", S("vLLM priority = critical path", rep(priority=True), "lor")),
        ("Scheduling", S("client limit 20 (KV-sized), critical path", rep(), "lor", 20)),
        ("Scheduling", S("client limit 20, finish teams in order", rep(), "lor", 20, None, True)),
        # Cache / KV
        ("Cache/KV", S("FP8 KV (2x tokens, 3,136-token blocks)", rep(kv_scale=2.0, block=3136), "lor")),
        ("Cache/KV", S("FP8 KV (2x tokens, blocks unchanged)", rep(kv_scale=2.0), "lor")),
        ("Cache/KV", S("LMCache CPU prefix cache", rep(lmcache=True), "lor")),
        ("Cache/KV", S("LMCache + team affinity", rep(lmcache=True), "affinity")),
        ("Cache/KV", S("fine-grained GPU prefix cache (256-token blocks)", rep(block=256), "affinity")),
        # Decode
        ("Decode", S("spec decode always on (MTP)", rep(spec="on"), "lor")),
        ("Decode", S("spec decode only at <=8 running", rep(spec="adaptive"), "lor")),
        # HyperQwen and GPU unlock
        ("HyperQwen/unlock", S("HyperQwen kernels only", rep(**HQ), "lor")),
        ("HyperQwen/unlock", S("HyperQwen kernels + adaptive spec", rep(**HQ_SPEC), "lor")),
        ("HyperQwen/unlock", S("HyperQwen full (+fine prefix cache, affinity)", rep(**HQ_FULL), "affinity")),
        ("HyperQwen/unlock", S("GPU unlock (clocks) on 4 copies", rep(card=UNLOCKED_3090), "lor")),
        ("HyperQwen/unlock", S("GPU unlock (clocks + P2P) on TP4", tp(4, 1, card=UNLOCKED_3090, link=w.PCIE_P2P))),
        ("HyperQwen/unlock", S("TP4 + P2P + HyperQwen kernels + adaptive spec",
                               tp(4, 1, link=w.PCIE_P2P, batch_tokens=8192, **HQ_SPEC))),
        ("HyperQwen/unlock", S("TP4 + P2P + unlock + HyperQwen full",
                               tp(4, 1, link=w.PCIE_P2P, batch_tokens=8192, card=UNLOCKED_3090, **HQ_FULL))),
        ("HyperQwen/unlock", S("2x TP2 + P2P + unlock + HyperQwen full, affinity",
                               tp(2, 2, link=w.PCIE_P2P, card=UNLOCKED_3090, **HQ_FULL), "affinity")),
        ("Stacked", S("4 copies + unlock + HyperQwen full + FP8 KV, affinity",
                      rep(card=UNLOCKED_3090, kv_scale=2.0, **HQ_FULL), "affinity")),
        ("Stacked", S("4 copies + FP8 KV + LMCache + adaptive spec, affinity (stock kernels)",
                      rep(kv_scale=2.0, lmcache=True, spec="adaptive"), "affinity")),
    ]
    return out


def scenarios_3080():
    S = w.Scenario
    c = CARD_3080
    u = UNLOCKED_3080
    return [
        ("3080 20GB", S("4 copies, round-robin", rep(card=c))),
        ("3080 20GB", S("4 copies, FP8 KV", rep(card=c, kv_scale=2.0, block=3136), "lor")),
        ("3080 20GB", S("4 copies, FP8 KV + LMCache + adaptive spec", rep(card=c, kv_scale=2.0, block=3136,
                                                                         lmcache=True, spec="adaptive"), "lor")),
        ("3080 20GB", S("2x TP2, no P2P", tp(2, 2, card=c), "lor")),
        ("3080 20GB", S("2x TP2 + P2P", tp(2, 2, card=c, link=w.PCIE_P2P), "lor")),
        ("3080 20GB", S("2x TP2 + P2P, team affinity", tp(2, 2, card=c, link=w.PCIE_P2P), "affinity")),
        ("3080 20GB", S("TP4, no P2P (stock)", tp(4, 1, card=c))),
        ("3080 20GB", S("TP4 + P2P, tuned", tp(4, 1, card=c, link=w.PCIE_P2P, batch_tokens=8192))),
        ("3080 20GB", S("2x TP2 + P2P + unlock + HyperQwen full, affinity",
                        tp(2, 2, card=u, link=w.PCIE_P2P, **HQ_FULL), "affinity")),
        ("3080 20GB", S("TP4 + P2P + unlock + HyperQwen full",
                        tp(4, 1, card=u, link=w.PCIE_P2P, batch_tokens=8192, **HQ_FULL))),
        ("3080 20GB", S("4 copies + unlock + HyperQwen full + FP8 KV",
                        rep(card=u, kv_scale=2.0, **HQ_FULL), "affinity")),
    ]


def model_b(sc: w.Scenario) -> w.Scenario:
    """Same scenario under the alternative cache model."""
    servers = tuple(replace(s, cache_keep_idle=False,
                            eff=EFF_B if s.eff == EFF else s.eff * EFF_B / EFF,
                            comm_scale=COMM_B) for s in sc.servers)
    return replace(sc, servers=servers)


def _run(args):
    group, sc, teams, model = args
    if model == "B":
        sc = model_b(sc)
    seeds = range(5) if teams <= 24 else range(3)
    mean, sd, r0 = w.run_seeds(sc, teams, seeds=seeds)
    return {"group": group, "label": sc.label, "teams": teams, "model": model, "mean": mean, "sd": sd,
            "cache_pct": r0["cache_pct"], "preempt": r0["preemptions"], "peak": r0["peak_running"],
            "kv_blocks": r0["kv_blocks"], "measured": MEASURED.get((sc.label, teams))}


def main(cmd: str) -> None:
    jobs = []
    if cmd == "calibrate":
        for g, sc in scenarios_3090():
            if g == "Calibration":
                for teams in (24, 96):
                    for m in ("A", "B"):
                        jobs.append((g, sc, teams, m))
    else:
        for g, sc in scenarios_3090():
            for teams in (24, 96):
                for m in ("A", "B"):
                    jobs.append((g, sc, teams, m))
        for g, sc in scenarios_3080():
            for teams in (24, 96):
                for m in ("A", "B"):
                    jobs.append((g, sc, teams, m))
    with mp.Pool() as pool:
        res = pool.map(_run, jobs)
    print(json.dumps(res))
