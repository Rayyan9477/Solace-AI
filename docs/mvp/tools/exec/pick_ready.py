"""List dependency-ready tasks for a wave/phase and their mutex conflicts with running lanes.

Usage: python pick_ready.py [--wave 0] [--phase A]
Reads D:\\tmp\\mvp-plan\\schedule_dag.json (wave, phase, mutex, real_infra, height) and
docs/mvp/TRACKER.md (deps column, status column). A task is ready when its status is todo and
every dep in the tracker deps column is done. Conflicts: hard = in-progress/fixing lanes,
soft = review lanes (a fix round may still need the files).
"""
from __future__ import annotations

import argparse
import json
import re

DAG = r"D:\tmp\mvp-plan\schedule_dag.json"
TRACKER = r"D:\Repo\Contextual-Chatbot\docs\mvp\TRACKER.md"
ROW = re.compile(r"^\| \[(WS-\d\d\.[0-9A-Z]+)\]")


def tracker_rows() -> dict[str, dict]:
    rows = {}
    for line in open(TRACKER, encoding="utf-8"):
        m = ROW.match(line)
        if not m:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        deps = [d.strip() for d in cells[6].split(",") if d.strip().startswith("WS-")]
        status = cells[8].strip("`")
        rows[m.group(1)] = {"deps": deps, "status": status, "model": cells[4]}
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=0)
    ap.add_argument("--phase", default="A")
    a = ap.parse_args()
    dag = json.load(open(DAG, encoding="utf-8"))
    rows = tracker_rows()
    hard, soft, infra = {}, {}, []
    for t, r in rows.items():
        n = dag.get(t, {})
        if r["status"] in ("in-progress", "fixing"):
            for k in n.get("mutex", []):
                hard[k] = t
            if n.get("real_infra"):
                infra.append(t)
        elif r["status"] == "review":
            for k in n.get("mutex", []):
                soft[k] = t
    print(f"running real-infra lanes: {infra}")
    ready = []
    for t, n in dag.items():
        r = rows.get(t)
        if not r or r["status"] != "todo" or n.get("wave") != a.wave or n.get("phase") != a.phase:
            continue
        missing = [d for d in r["deps"] if rows.get(d, {}).get("status") != "done"]
        if missing:
            continue
        h = sorted({hard[k] for k in n.get("mutex", []) if k in hard})
        s = sorted({soft[k] for k in n.get("mutex", []) if k in soft})
        ready.append((n.get("height", 0), t, n.get("real_infra"), r["model"], h, s))
    for height, t, ri, model, h, s in sorted(ready, reverse=True):
        flag = "FREE" if not h and not s else ("SOFT" if not h else "BLOCKED")
        print(f"{flag:7} {t:10} h={height:<3} infra={str(ri):5} model={model:7} hard={h} soft={s}")
    blocked = [t for t, n in dag.items() if rows.get(t, {}).get("status") == "todo"
               and n.get("wave") == a.wave and n.get("phase") == a.phase and (t, ) not in [(x[1],) for x in ready]]
    print(f"waiting on deps: {len(blocked)} {blocked}")


if __name__ == "__main__":
    main()
