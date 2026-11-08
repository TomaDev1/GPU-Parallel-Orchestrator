"""Simulator presets for the target hardware.

These are ESTIMATES for developing without the hardware, not measurements.
Replace them with real numbers once the workstation is running.

``QWEN27B_RTX3080_20GB``: Qwen3.8-27B at 4-bit, one copy per RTX 3080 20GB.

* Decode speed: third-party RTX 3090 vLLM reports for a 27B 4-bit model
  (~46 tok/s for one request, ~1,035 tok/s total at 64 concurrent), scaled
  by ~0.81 for the 3080's lower memory bandwidth (~760 vs ~936 GB/s).
* Prefill: ~1,800 tok/s on a 3090, scaled to ~1,450 tok/s.
* KV cache: ~15-16GB of weights leaves roughly 3-4GB. At an assumed
  ~128KB per token (FP8 KV cache) that's ~30k tokens in flight per card.
  With BF16 KV, halve it.
"""

from __future__ import annotations

QWEN27B_RTX3080_20GB: dict = {
    "speed_curve": {1: 37.0, 64: 13.1},
    "prefill_tps": 1450.0,
    "kv_capacity_tokens": 30_000,
    "max_batch": 0,
}
