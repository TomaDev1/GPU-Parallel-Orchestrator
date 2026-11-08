"""Run a small task graph against a real OpenAI-compatible server.

    pip install -e .[http]
    python examples/demo_server.py --base-url http://localhost:8000/v1 --model qwen --concurrency 8

Works with vLLM, SGLang, llama.cpp server, LM Studio, or any hosted
OpenAI-compatible API (pass --api-key).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchestrator import OpenAICompatibleBackend, Scheduler, Task, TaskGraph  # noqa: E402

TOPICS = ["CPU", "GPU", "memory", "storage", "cooling", "power supply"]


def build_graph() -> TaskGraph:
    g = TaskGraph()
    for topic in TOPICS:
        g.add(Task(f"eval_{topic}", f"In 5 bullet points, what matters when choosing a {topic} "
                   "for a 4-GPU LLM inference workstation?", max_tokens=300))
    g.add(Task(
        "summary",
        lambda deps: "Combine these notes into a one-paragraph buying guide:\n\n"
        + "\n\n".join(deps.values()),
        deps=tuple(f"eval_{t}" for t in TOPICS),
        max_tokens=400,
    ))
    return g


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    backend = OpenAICompatibleBackend(args.base_url, args.model, api_key=args.api_key)
    try:
        graph = build_graph()
        report = await Scheduler(backend, max_concurrency=args.concurrency).run(graph)
        print(report.summary())
        print("\n" + (graph["summary"].output or graph["summary"].error or ""))
    finally:
        await backend.aclose()


if __name__ == "__main__":
    asyncio.run(main())
