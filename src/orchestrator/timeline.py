"""Render a run as a self-contained HTML timeline.

    from orchestrator.timeline import write_timeline
    write_timeline(report, "timeline.html", title="Research batch")

One lane per server (endpoint). Within a lane, each bar is a task from start
to finish, packed into rows so overlapping tasks are visible. A thin line in
front of a bar is time the task spent ready but waiting for a slot. Tasks on
the run's critical path are highlighted: that chain decided when the run
finished, so shortening or splitting those tasks is what makes the whole
job faster.

No external files or scripts: the page is a single HTML file with inline SVG.
"""

from __future__ import annotations

import html
from pathlib import Path

from .graph import TaskState
from .scheduler import RunReport

ROW_H = 7
ROW_GAP = 2
LANE_GAP = 22
LEFT = 70
WIDTH = 1100


def path_breakdown(report: RunReport) -> dict:
    """How the critical path's time splits into running vs waiting for a slot."""
    path = report.critical_path()
    run_time = sum(report.records[t].duration or 0.0 for t in path)
    wait_time = sum(report.records[t].wait or 0.0 for t in path)
    return {
        "tasks": path,
        "running": run_time,
        "waiting": wait_time,
        "makespan": report.makespan,
    }


def _pack(intervals: list[tuple[float, float, str]]) -> list[list[tuple[float, float, str]]]:
    """Greedy interval packing into rows (first row whose last bar has ended)."""
    rows: list[list[tuple[float, float, str]]] = []
    ends: list[float] = []
    for iv in sorted(intervals):
        for i, end in enumerate(ends):
            if iv[0] >= end:
                rows[i].append(iv)
                ends[i] = iv[1]
                break
        else:
            rows.append([iv])
            ends.append(iv[1])
    return rows


def render_timeline(report: RunReport, title: str = "Run timeline") -> str:
    recs = {k: r for k, r in report.records.items() if r.started_at is not None and r.finished_at is not None}
    t0 = report.started_at
    span = max(report.makespan, 1e-9)
    crit = set(report.critical_path())
    scale = (WIDTH - LEFT - 10) / span

    def x(t: float) -> float:
        return LEFT + (t - t0) * scale

    lanes: dict[str, list] = {}
    for r in recs.values():
        # Pack from when the task became ready, so its wait line has room.
        begin = min(r.ready_at, r.started_at) if r.ready_at is not None else r.started_at
        lanes.setdefault(r.endpoint or "backend", []).append((begin, r.finished_at, r.task_id))

    parts: list[str] = []
    y = 28
    busy_rows = []
    for lane in sorted(lanes):
        rows = _pack(lanes[lane])
        busy = sum(recs[t].finished_at - recs[t].started_at for _, _, t in lanes[lane])
        busy_rows.append((lane, len(lanes[lane]), busy / span))
        parts.append(f'<text x="4" y="{y + 10}" class="lane">{html.escape(lane)}</text>')
        for row in rows:
            for _, e, tid in row:
                r = recs[tid]
                s = r.started_at
                cls = "bar crit" if tid in crit else "bar"
                if r.state is TaskState.FAILED:
                    cls += " failed"
                tip = (
                    f"{tid}\nstart {s - t0:.2f}s, took {e - s:.2f}s"
                    f"\nwaited {r.wait or 0:.2f}s for a slot"
                    f"\n{r.output_tokens} output tokens, {r.cached_tokens} prompt tokens from cache"
                )
                if r.ready_at is not None and r.ready_at < s:
                    parts.append(
                        f'<line x1="{x(r.ready_at):.1f}" x2="{x(s):.1f}" y1="{y + ROW_H / 2:.1f}" '
                        f'y2="{y + ROW_H / 2:.1f}" class="wait{" crit" if tid in crit else ""}"/>'
                    )
                w = max(1.0, (e - s) * scale)
                parts.append(
                    f'<rect x="{x(s):.1f}" y="{y}" width="{w:.1f}" height="{ROW_H}" rx="1.5" class="{cls}">'
                    f"<title>{html.escape(tip)}</title></rect>"
                )
            y += ROW_H + ROW_GAP
        y += LANE_GAP

    # Time axis.
    ticks = []
    step = _nice_step(span)
    t = 0.0
    while t <= span + 1e-9:
        ticks.append(
            f'<line x1="{x(t0 + t):.1f}" x2="{x(t0 + t):.1f}" y1="18" y2="{y - LANE_GAP + 4}" class="grid"/>'
            f'<text x="{x(t0 + t):.1f}" y="12" class="tick">{t:g}s</text>'
        )
        t += step
    height = y

    pb = path_breakdown(report)
    waits = [r.wait for r in recs.values() if r.wait is not None]
    avg_wait = sum(waits) / len(waits) if waits else 0.0
    lane_rows = "".join(
        f"<tr><td>{html.escape(n)}</td><td>{c}</td><td>{u:.1f}</td></tr>" for n, c, u in busy_rows
    )
    path_list = " &rarr; ".join(html.escape(t) for t in pb["tasks"])

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ --bg:#fbfaf8; --fg:#1d1d1f; --muted:#6b6b70; --bar:#8aa4c8; --crit:#d9653b;
        --wait:#b9b9bf; --grid:#e6e4df; --fail:#b3261e; --card:#ffffff; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#17171a; --fg:#ececef; --muted:#9a9aa2; --bar:#5d7ba6; --crit:#f08a5d;
          --wait:#55555c; --grid:#2a2a2f; --fail:#f2b8b5; --card:#1f1f23; }}
}}
body {{ margin:0; padding:24px 16px; background:var(--bg); color:var(--fg);
       font:14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; }}
main {{ max-width:{WIDTH + 40}px; margin:0 auto; }}
h1 {{ font-size:20px; margin:0 0 4px; }}
.sub {{ color:var(--muted); margin:0 0 16px; }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:8px; margin-bottom:16px; }}
.stat {{ background:var(--card); border:1px solid var(--grid); border-radius:8px; padding:10px 12px; }}
.stat b {{ display:block; font-size:18px; }}
.stat span {{ color:var(--muted); font-size:12px; }}
.chart {{ overflow-x:auto; background:var(--card); border:1px solid var(--grid); border-radius:8px; padding:8px; }}
svg {{ display:block; }}
.bar {{ fill:var(--bar); }} .bar.crit {{ fill:var(--crit); }} .bar.failed {{ fill:var(--fail); }}
.wait {{ stroke:var(--wait); stroke-width:1; }} .wait.crit {{ stroke:var(--crit); stroke-dasharray:2 2; }}
.grid {{ stroke:var(--grid); }} .tick {{ fill:var(--muted); font-size:10px; text-anchor:middle; }}
.lane {{ fill:var(--fg); font-size:12px; font-weight:600; }}
.legend {{ color:var(--muted); font-size:12px; margin:8px 0 16px; }}
.legend i {{ display:inline-block; width:12px; height:8px; border-radius:2px; margin:0 4px 0 12px; vertical-align:middle; }}
table {{ border-collapse:collapse; font-size:13px; }}
td, th {{ padding:4px 12px 4px 0; text-align:left; }} th {{ color:var(--muted); font-weight:500; }}
.path {{ font-size:12px; color:var(--muted); word-break:break-word; }}
</style></head>
<body><main>
<h1>{html.escape(title)}</h1>
<p class="sub">{len(report.records)} tasks &middot; hover a bar for details</p>
<div class="stats">
  <div class="stat"><b>{report.makespan:.1f}s</b><span>total time</span></div>
  <div class="stat"><b>{report.throughput:,.0f}</b><span>output tok/s overall</span></div>
  <div class="stat"><b>{report.peak_in_flight}</b><span>peak requests in flight</span></div>
  <div class="stat"><b>{avg_wait:.2f}s</b><span>average wait for a slot</span></div>
  <div class="stat"><b>{pb['running']:.1f}s</b><span>critical path running</span></div>
  <div class="stat"><b>{pb['waiting']:.1f}s</b><span>critical path waiting for slots</span></div>
</div>
<div class="legend"><i style="background:var(--bar)"></i>task
<i style="background:var(--crit)"></i>critical path
<i style="background:var(--wait);height:2px"></i>waiting for a slot
<i style="background:var(--fail)"></i>failed</div>
<div class="chart"><svg width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" role="img"
 aria-label="Timeline of tasks per server">{''.join(ticks)}{''.join(parts)}</svg></div>
<h2 style="font-size:15px;margin:20px 0 6px">Per server</h2>
<table><tr><th>Server</th><th>Tasks</th><th>Avg requests in flight</th></tr>{lane_rows}</table>
<h2 style="font-size:15px;margin:20px 0 6px">Critical path ({len(pb['tasks'])} tasks)</h2>
<p class="path">{path_list}</p>
</main></body></html>
"""


def _nice_step(span: float) -> float:
    raw = span / 10
    mag = 10 ** int(f"{raw:e}".split("e")[1])
    for m in (1, 2, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def write_timeline(report: RunReport, path: str | Path, title: str = "Run timeline") -> Path:
    path = Path(path)
    path.write_text(render_timeline(report, title), encoding="utf-8")
    return path
