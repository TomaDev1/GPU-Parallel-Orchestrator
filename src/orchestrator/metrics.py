"""Persistent metrics: runs, per-task timings, controller history, learned curves.

Everything goes into one SQLite file (stdlib only):

    store = MetricsStore("metrics.db")
    pool = BackendPool([...Endpoint(name, backend, controller=store.controller(name))...])
    report = await Scheduler(pool, ...).run(graph)
    run_id = store.save_run(report, pool, label="nightly research batch")

``store.controller(name)`` returns an AdaptiveLimit warm-started from the
curve that endpoint learned last time, so only the very first run pays for
probing. ``save_run`` stores the run and updates the learned curves.

Endpoint names are the key for curves, so keep them stable (for example
"gpu0".."gpu3") and change them if you change the model or quantisation.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .admission import AdaptiveLimit
from .scheduler import RunReport

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT,
    recorded_at REAL NOT NULL,
    makespan REAL NOT NULL,
    throughput REAL NOT NULL,
    peak_in_flight INTEGER NOT NULL,
    tasks INTEGER NOT NULL,
    succeeded INTEGER NOT NULL,
    failed INTEGER NOT NULL,
    critical_path TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    task_id TEXT NOT NULL,
    state TEXT NOT NULL,
    parent TEXT,
    deps TEXT NOT NULL,
    endpoint TEXT,
    ready_at REAL,
    started_at REAL,
    finished_at REAL,
    wait REAL,
    prompt_tokens INTEGER,
    output_tokens INTEGER,
    cached_tokens INTEGER,
    error TEXT,
    PRIMARY KEY (run_id, task_id)
);
CREATE TABLE IF NOT EXISTS limit_history (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    endpoint TEXT NOT NULL,
    at REAL NOT NULL,
    "limit" INTEGER NOT NULL,
    throughput REAL NOT NULL,
    phase TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS curves (
    endpoint TEXT PRIMARY KEY,
    curve TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


class MetricsStore:
    def __init__(self, path: str | Path = "metrics.db") -> None:
        self.path = str(path)
        self._db = sqlite3.connect(self.path)
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "MetricsStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- learned curves ---------------------------------------------------

    def load_curve(self, endpoint: str) -> dict[int, float] | None:
        row = self._db.execute("SELECT curve FROM curves WHERE endpoint = ?", (endpoint,)).fetchone()
        if row is None:
            return None
        return {int(k): float(v) for k, v in json.loads(row[0]).items()}

    def save_curve(self, endpoint: str, curve: dict[int, float]) -> None:
        if not curve:
            return
        self._db.execute(
            "INSERT INTO curves (endpoint, curve, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(endpoint) DO UPDATE SET curve = excluded.curve, updated_at = excluded.updated_at",
            (endpoint, json.dumps({str(k): v for k, v in sorted(curve.items())}), time.time()),
        )
        self._db.commit()

    def controller(self, endpoint: str, **kwargs: Any) -> AdaptiveLimit:
        """An AdaptiveLimit warm-started from this endpoint's saved curve, if any."""
        kwargs.setdefault("initial_curve", self.load_curve(endpoint))
        return AdaptiveLimit(**kwargs)

    # -- runs -------------------------------------------------------------

    def save_run(self, report: RunReport, pool=None, label: str | None = None) -> int:
        """Store a run (and, if a pool is given, its controllers' history and curves)."""
        from .graph import TaskState

        cur = self._db.execute(
            "INSERT INTO runs (label, recorded_at, makespan, throughput, peak_in_flight, tasks, "
            "succeeded, failed, critical_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                label,
                time.time(),
                report.makespan,
                report.throughput,
                report.peak_in_flight,
                len(report.records),
                report.count(TaskState.SUCCEEDED),
                report.count(TaskState.FAILED),
                json.dumps(report.critical_path()),
            ),
        )
        run_id = cur.lastrowid
        self._db.executemany(
            "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    run_id, r.task_id, r.state.value, r.parent, json.dumps(list(r.deps)),
                    r.endpoint, r.ready_at, r.started_at, r.finished_at, r.wait,
                    r.prompt_tokens, r.output_tokens, r.cached_tokens, r.error,
                )
                for r in report.records.values()
            ],
        )
        for ep in getattr(pool, "endpoints", []) or []:
            ctrl = getattr(ep, "controller", None)
            if ctrl is None:
                continue
            self._db.executemany(
                'INSERT INTO limit_history (run_id, endpoint, at, "limit", throughput, phase) '
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(run_id, ep.name, h.at, h.limit, h.throughput, h.phase) for h in ctrl.history],
            )
            self.save_curve(ep.name, ctrl.curve)
        self._db.commit()
        return run_id

    def runs(self, limit: int = 20) -> list[dict]:
        cur = self._db.execute(
            "SELECT id, label, recorded_at, makespan, throughput, tasks, succeeded, failed "
            "FROM runs ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def run_tasks(self, run_id: int) -> list[dict]:
        cur = self._db.execute("SELECT * FROM tasks WHERE run_id = ? ORDER BY started_at", (run_id,))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
