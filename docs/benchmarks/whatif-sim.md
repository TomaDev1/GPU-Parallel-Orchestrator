# What-if simulations on top of the 4× RTX 3090 bench (2026-11-09)

The Vast.ai box is shut down, so these are **simulations, not measurements**.
They ask one question: which changes could plausibly beat the measured
4-copy result (177 s for 24 teams, 653 s for 96 teams), by how much, and
which are worth paying for real GPU time to test.

Simulator: [`scripts/whatif_sim.py`](../../scripts/whatif_sim.py), scenarios in
[`scripts/whatif_report.py`](../../scripts/whatif_report.py). Reproduce with
`python scripts/whatif_sim.py report` (about 5 minutes on 4 cores).

## Summary

| Lever | Est. time saved vs 4 copies | Confidence | Test it? |
| --- | --- | --- | --- |
| **Make the prefix cache hit the whole team brief** (LMCache, or a vLLM/HyperQwen build that caches in small blocks) | 25–45% | Low–medium: biggest lever in the model, but caching is exactly what the last simulation got wrong | **Yes, first** |
| **2× TP2 with P2P enabled** | 14–27% on 3090s; **~60% on the 3080 build** | Medium on 3080s (memory math), low on 3090s (P2P speed assumed) | **Yes** |
| **HyperQwen patch set** (kernels + spec decoding at low load) | 15–23% | Low–medium: assumes their published speeds carry over | **Yes** |
| **TP4 with P2P** (patched driver) | 2–12% on 3090s (close to a tie) | Low: P2P link speed is assumed, not measured | Yes, if P2P is enabled anyway |
| FP8 KV cache | −3% to +20% on 3090s, ~23% on 3080s | Low: depends on whether vLLM doubles the hybrid block size | Yes, check the block size in the log |
| GPU unlock: power limit + memory overclock | 7–12% (≈ the memory-clock gain) | Medium | Cheap; do it if thermals allow |
| Speculative decoding only when ≤8 requests run | 2–12% (most on small jobs) | Low–medium | Yes, but **never always-on**: −10% on 96 teams |
| Team-affinity routing | 11–22% in the model | **Low**: the model over-predicted this before | Only together with a cache fix |
| Least-outstanding-requests routing | 1–6% | Medium | Cheap; make it the default |
| Dispatch teams in order (finish a team before starting new ones) | 2–8% | Low–medium | Cheap |
| vLLM priority = critical path | −5% to +9% | Low | No |
| Client-side concurrency limits | about 0% or worse on 96 teams | Medium (matches the real runs) | No |

**Do they stack?** Mostly yes, because they hit different costs: the prefix
cache removes prefill work (~40% of GPU time in this workload), kernels and
clocks speed every step, speculative decoding speeds the low-concurrency
tail, and the layout sets how much KV memory there is. A full stack
(unlock + HyperQwen + fine-grained prefix cache, on 4 copies or 2× TP2+P2P)
comes out at **about 80–100 s for 24 teams and 250–340 s for 96 teams** in
the model, roughly half the measured time. Treat that as an upper bound:
it multiplies several uncertain gains.

**For the 4× 3080 20GB build** the layout question dominates: with this
model, 4 copies leave ~0.8 GiB of KV per card (3–4 requests), and the model
puts them at ~400 s for 24 teams. **2× TP2 is ~1.9× faster than 4 copies
even without P2P** (209 s), and ~2.5× with P2P (147–162 s). Stock TP4 without
P2P lands in between (275 s).

## How the model works and how well it fits

It simulates vLLM step by step on each server: chunked prefill and decode
sharing a 2,048-token step budget, a block KV cache with the hybrid model's
1,568-token blocks, FCFS admission, preemption by recompute when KV runs
out, prefix caching only in whole blocks, and tensor parallelism as a
per-step all-reduce cost. Per-card speeds start from the published RTX 3090
figures for this model (46 tok/s single stream, ~300 at 8, ~1,035 at 64,
~1,800 tok/s prefill). Two factors were then fitted to the real runs:

- **Stock vLLM is 1.15× slower** than those published figures (fitted to run
  10, 4 copies, 177.1 s).
- **The no-P2P all-reduce** works out to ~75 µs per call and ~5 GB/s
  effective (fitted to run 9, TP4, 257.6 s).

Everything else is a held-out check:

| Run | Measured | Model A | Model B |
| --- | ---: | ---: | ---: |
| 24 teams, limit 4 | 241.2 s | 242 s | 261 s |
| 24 teams, limit 8 | 192.2 s | 182 s | 199 s |
| 24 teams, limit 16 | 195.4 s | **150 s** | **164 s** |
| 24 teams, limit 32 | 187.2 s | **148 s** | **160 s** |
| 96 teams, 4 copies unmanaged | 652.6 s | 656 s | 632 s |
| 96 teams, limit 40 + prefix routing | 666.2 s | 674 s | 684 s |
| Prefix-cache hits, 24 / 96 / 96 managed | 12% / 11% / 7% | 19% / 11% / 17% | 9% / 12% / 10% |

**Calibration error, stated plainly:** the big runs and the low limits fit
within ~6%, but the model is **15–23% too optimistic about the limit 16
and 32 runs**, and I couldn't find why. Model A keeps idle prefixes cached
(as vLLM does); model B drops a prefix once no request uses it, which fits
the measured hit rates better but the low limits worse. Every result below
is shown under both, and the spread between them is a rough error bar. In
the tables, a gain from client-side routing or limits should be read as an
upper bound, since that is where the model has been wrong.

Run-to-run noise in the model (5 seeds, ±4% step jitter) is ±1–4 s at 24
teams and up to ±19 s at 96 teams. The real 177 s vs 183 s difference is inside
that.

## Results (4× RTX 3090)

"A / B" are the two cache models. Time saved is against 4 copies with
round-robin routing in the same model (174 s / 175 s at 24 teams, 656 s /
632 s at 96 teams); negative means slower.
### Layout

| Scenario | 24 teams (A / B) | time saved | 96 teams (A / B) | time saved |
| --- | ---: | ---: | ---: | ---: |
| TP4 + P2P | 153s / 164s | +6% to +12% | 607s / 620s | +2% to +8% |
| TP4 tuned (8k batch tokens), no P2P | 257s / 263s | -51% to -48% | 1020s / 980s | -55% |
| TP4 + P2P, tuned | 153s / 168s | +4% to +12% | 607s / 625s | +1% to +8% |
| 2x TP2, no P2P, least-outstanding | 189s / 191s | -9% | 703s / 662s | -7% to -5% |
| 2x TP2 + P2P, least-outstanding | 139s / 150s | +14% to +20% | 522s / 525s | +17% to +20% |
| 2x TP2 + P2P, team affinity | 127s / 139s | +21% to +27% | 476s / 498s | +21% to +27% |

### Routing

| Scenario | 24 teams (A / B) | time saved | 96 teams (A / B) | time saved |
| --- | ---: | ---: | ---: | ---: |
| 4 copies, least-outstanding requests | 163s / 169s | +4% to +6% | 637s / 624s | +1% to +3% |
| 4 copies, least-outstanding tokens | 159s / 169s | +3% to +8% | 639s / 620s | +2% to +3% |
| 4 copies, team affinity | 135s / 151s | +14% to +22% | 577s / 561s | +11% to +12% |

### Scheduling

| Scenario | 24 teams (A / B) | time saved | 96 teams (A / B) | time saved |
| --- | ---: | ---: | ---: | ---: |
| vLLM priority = critical path | 158s / 165s | +5% to +9% | 686s / 666s | -5% |
| client limit 20 (KV-sized), critical path | 151s / 162s | +7% to +13% | 663s / 651s | -3% to -1% |
| client limit 20, finish teams in order | 162s / 171s | +2% to +7% | 604s / 604s | +4% to +8% |

### Cache and KV

| Scenario | 24 teams (A / B) | time saved | 96 teams (A / B) | time saved |
| --- | ---: | ---: | ---: | ---: |
| FP8 KV (2x tokens, 3,136-token blocks) | 179s / 174s | -3% to +1% | 634s / 614s | +3% |
| FP8 KV (2x tokens, blocks unchanged) | 146s / 163s | +7% to +16% | 520s / 519s | +18% to +21% |
| LMCache CPU prefix cache | 108s / 104s | +38% to +40% | 364s / 353s | +44% to +45% |
| LMCache + team affinity | 108s / 105s | +38% to +40% | 358s / 352s | +44% to +45% |
| fine-grained GPU prefix cache (256-token blocks) | 101s / 129s | +26% to +42% | 409s / 399s | +37% to +38% |

### Decode

| Scenario | 24 teams (A / B) | time saved | 96 teams (A / B) | time saved |
| --- | ---: | ---: | ---: | ---: |
| spec decode always on (MTP) | 172s / 180s | -3% to +1% | 721s / 705s | -12% to -10% |
| spec decode only at <=8 running | 153s / 161s | +8% to +12% | 634s / 619s | +2% to +3% |

### HyperQwen and GPU unlock

| Scenario | 24 teams (A / B) | time saved | 96 teams (A / B) | time saved |
| --- | ---: | ---: | ---: | ---: |
| HyperQwen kernels only | 141s / 146s | +16% to +19% | 553s / 543s | +14% to +16% |
| HyperQwen kernels + adaptive spec | 133s / 140s | +20% to +23% | 551s / 538s | +15% to +16% |
| HyperQwen full (+fine prefix cache, affinity) | 83s / 107s | +39% to +52% | 351s / 343s | +46% to +47% |
| GPU unlock (clocks) on 4 copies | 153s / 159s | +9% to +12% | 604s / 589s | +7% to +8% |
| GPU unlock (clocks + P2P) on TP4 | 149s / 159s | +9% to +14% | 590s / 601s | +5% to +10% |
| TP4 + P2P + HyperQwen kernels + adaptive spec | 142s / 154s | +12% to +18% | 563s / 574s | +9% to +14% |
| TP4 + P2P + unlock + HyperQwen full | 94s / 139s | +20% to +46% | 349s / 471s | +26% to +47% |
| 2x TP2 + P2P + unlock + HyperQwen full, affinity | 79s / 102s | +42% to +55% | 256s / 342s | +46% to +61% |

### Stacked

| Scenario | 24 teams (A / B) | time saved | 96 teams (A / B) | time saved |
| --- | ---: | ---: | ---: | ---: |
| 4 copies + unlock + HyperQwen full + FP8 KV, affinity | 78s / 102s | +42% to +55% | 253s / 318s | +50% to +61% |
| 4 copies + FP8 KV + LMCache + adaptive spec, affinity (stock kernels) | 95s / 92s | +45% to +47% | 306s / 308s | +51% to +53% |

## Projection: 4× RTX 3080 20GB

Time saved is against 4 copies on the 3080s in the same model.

| Scenario | 24 teams (A / B) | time saved | 96 teams (A / B) | time saved |
| --- | ---: | ---: | ---: | ---: |
| 4 copies, round-robin | 405s / 397s | +0% | 1607s / 1615s | +0% |
| 4 copies, FP8 KV | 310s / 301s | +23% to +24% | 1255s / 1216s | +22% to +25% |
| 4 copies, FP8 KV + LMCache + adaptive spec | 177s / 171s | +56% to +57% | 711s / 691s | +56% to +57% |
| 2x TP2, no P2P | 209s / 216s | +46% to +48% | 894s / 810s | +44% to +50% |
| 2x TP2 + P2P | 162s / 175s | +56% to +60% | 689s / 659s | +57% to +59% |
| 2x TP2 + P2P, team affinity | 147s / 162s | +59% to +64% | 641s / 633s | +60% to +61% |
| TP4, no P2P (stock) | 275s / 279s | +30% to +32% | 1094s / 1048s | +32% to +35% |
| TP4 + P2P, tuned | 170s / 188s | +52% to +58% | 685s / 707s | +56% to +57% |
| 2x TP2 + P2P + unlock + HyperQwen full, affinity | 91s / 118s | +70% to +77% | 301s / 401s | +75% to +81% |
| TP4 + P2P + unlock + HyperQwen full | 104s / 156s | +61% to +74% | 386s / 525s | +68% to +76% |
| 4 copies + unlock + HyperQwen full + FP8 KV | 126s / 146s | +63% to +69% | 526s / 538s | +67% |

## What the scenarios assume

- **P2P**: all-reduce at 15 µs per call and 20 GB/s (PCIe 4.0 x16 with
  peer-to-peer, as with the patched driver on cards with Resizable BAR), vs
  the fitted ~75 µs and ~5 GB/s without it. This is a ~4× cheaper all-reduce.
  It's the main assumption behind every P2P row; a commenter's 4×3090 TP4
  with P2P and a PCIe 4.0 switch came out best on their box, which is
  consistent with this, but I couldn't measure it.
- **HyperQwen kernels**: the published per-card speeds with no stock-vLLM
  penalty, i.e. ~15% faster steps. "Full" adds speculative decoding at ≤8
  running requests and prefix caching in 256-token blocks, since their
  shared-prompt test (222 s → 16.9 s) shows this model can cache well.
- **Speculative decoding**: MTP with 3 draft tokens, 2.4 tokens accepted per
  step, cost tuned so it stops paying off at ~8 concurrent requests (as
  reported for a 3090). "Adaptive" switches it off above 8, which stock
  vLLM may not do; check before relying on it.
- **LMCache**: a CPU-RAM prefix cache shared by all servers on the host,
  256-token chunks, loaded at 12 GB/s. Whether it supports this hybrid
  (Mamba-state) model in vLLM 0.32 is **not verified**; that's the first
  thing to check.
- **FP8 KV**: 2× the KV tokens. If vLLM keeps hybrid pages equal in size, the
  block becomes 3,136 tokens, longer than any team brief, so prefix caching
  stops working (the −3% to +3% row). If the block size stays at 1,568, it's
  +7% to +21%.
- **GPU unlock**: +7% memory bandwidth and +5% compute from a raised power
  limit and memory overclock. Decode is memory-bound, so the gain tracks the
  memory clock; stability and thermals are up to the cards.
- **3080 20GB**: 760 vs 936 GB/s bandwidth, 0.84× compute, ~0.8 GiB KV per
  card for 4 copies (from the earlier estimate), same PCIe link assumptions.

## What to test on real hardware, in order

1. **Is the prefix cache the real bottleneck?** Re-run the 96-team bench
   with longer briefs (≥3 blocks, ~5k tokens) or with LMCache if it loads
   this model, and read `cached_tokens`. If hits go from ~11% to 50%+ and the
   makespan drops, the biggest lever is real. Also read
   `vllm:num_preemptions_total`: the model says ~200 preemptions and ~20% of
   prefill wasted on the 96-team run.
2. **2× TP2 vs 4 copies**, with and without the P2P driver. This is the main
   layout decision for the 3080 build.
3. **HyperQwen's patch set as the backend**, same bench, to see whether its
   published speeds hold on this workload.
4. **TP4 with P2P** if the P2P driver is installed anyway; expect about a
   tie with 4 copies on 3090s.
5. **FP8 KV**: check the "attention block size" line in the vLLM log first.
6. Cheap coordinator changes: default to least-outstanding-requests
   routing, and optionally dispatch teams in order.

Repeat each config at least 3 times with a warm-up run discarded, and tune
both layouts with the same effort, as the reviewers suggested.
