"""End-to-end check of scripts/bench.py against fake OpenAI-compatible servers."""

import asyncio
import importlib.util
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("httpx")

BENCH = Path(__file__).resolve().parents[1] / "scripts" / "bench.py"


class FakeOpenAI(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"data": [{"id": "fake-qwen"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert req["model"] == "fake-qwen"
        time.sleep(0.005)
        self._send({
            "choices": [{"message": {"content": "1. Intro\n2. Details"}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": min(req["max_tokens"], 20)},
        })


@pytest.fixture
def servers():
    started = []
    for _ in range(2):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAI)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        started.append(srv)
    yield [f"http://127.0.0.1:{s.server_address[1]}/v1" for s in started]
    for s in started:
        s.shutdown()


def load_bench():
    spec = importlib.util.spec_from_file_location("bench", BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bench_runs_against_real_http_servers(servers, tmp_path, monkeypatch, capsys):
    bench = load_bench()
    monkeypatch.setattr(sys, "argv", [
        "bench.py", "--endpoints", ",".join(servers), "--teams", "3", "--limits", "4",
        "--adaptive-runs", "1", "--db", str(tmp_path / "b.db"), "--out", str(tmp_path / "out"),
    ])
    asyncio.run(bench.main())
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    labels = [r["label"] for r in summary["runs"]]
    assert labels == ["fixed 4", "adaptive #1", "learned, frozen"]
    assert all(r["tasks"] > 3 for r in summary["runs"])  # plan tasks spawned subtasks
    assert (tmp_path / "out" / "timeline-run1.html").exists()
    assert "task(s) failed" not in capsys.readouterr().out


def test_bench_simulate_mode(tmp_path, monkeypatch):
    bench = load_bench()
    monkeypatch.setattr(sys, "argv", [
        "bench.py", "--simulate", "--teams", "2", "--adaptive-runs", "1", "--time-scale", "5000",
        "--db", str(tmp_path / "b.db"), "--out", str(tmp_path / "out"),
    ])
    asyncio.run(bench.main())
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["runs"]


def test_bench_unmanaged_baseline(servers, tmp_path, monkeypatch):
    bench = load_bench()
    monkeypatch.setattr(sys, "argv", [
        "bench.py", "--endpoints", ",".join(servers), "--teams", "3", "--unmanaged", "--adaptive-runs", "0",
        "--db", str(tmp_path / "b.db"), "--out", str(tmp_path / "out"),
    ])
    asyncio.run(bench.main())
    runs = json.loads((tmp_path / "out" / "summary.json").read_text())["runs"]
    assert [r["label"] for r in runs] == ["unmanaged"]
    assert set(runs[0]["limits"].values()) == {"none"}


def test_bench_reports_prefix_cache_rate_and_sim_batch_slots(tmp_path, monkeypatch, capsys):
    bench = load_bench()
    monkeypatch.setattr(sys, "argv", [
        "bench.py", "--simulate", "--teams", "2", "--limits", "8", "--adaptive-runs", "0",
        "--sim-max-batch", "4", "--time-scale", "5000",
        "--db", str(tmp_path / "b.db"), "--out", str(tmp_path / "out"),
    ])
    asyncio.run(bench.main())
    run = json.loads((tmp_path / "out" / "summary.json").read_text())["runs"][0]
    assert 0 < run["prefix_cache_hit_rate"] < 1
    assert "prefix cache" in capsys.readouterr().out
