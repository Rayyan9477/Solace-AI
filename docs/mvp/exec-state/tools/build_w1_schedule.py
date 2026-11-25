"""Build the W1 dispatch table: schedule_dag.json + TRACKER.md + the W1 pre-flight rulings' extra deps."""
import json
import re
import sys
from pathlib import Path

REPO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
dag = json.loads((REPO / "docs/mvp/exec-state/plan/schedule_dag.json").read_text(encoding="utf-8"))
ROW = re.compile(r"^\| \[(WS-\d\d\.[0-9A-Z]+)\]")
rows = {}
for line in (REPO / "docs/mvp/TRACKER.md").read_text(encoding="utf-8").splitlines():
    m = ROW.match(line)
    if not m:
        continue
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    rows[m.group(1)] = {
        "title": c[1], "severity": c[2], "review": c[3], "model": c[4].lower(),
        "deps": [d.strip() for d in c[6].split(",") if d.strip().startswith("WS-")],
        "status": c[8].strip("`"),
    }

# Extra dependencies from docs/mvp/exec-state/plan/W1-preflight-scan.md part 3.
EXTRA = {
    "WS-04.6": (["WS-06.8", "WS-07.7", "WS-07.3"], "R-3"),
    "WS-07.2": (["WS-07.3"], "R-4"),
    "WS-07.4": (["WS-07.2"], "R-4"),
    "WS-07.5": (["WS-07.4"], "R-4"),
    "WS-05.21": (["WS-03.19", "WS-09.18", "WS-09.4", "WS-07.7"], "R-5"),
    "WS-05.8": (["WS-05.21"], "R-5"),
    "WS-05.9": (["WS-05.8"], "R-5"),
    "WS-05.22": (["WS-05.9", "WS-12.3"], "R-5"),
    "WS-05.20": (["WS-09.3", "WS-07.3"], "R-7"),
    "WS-13.13": (["WS-10.3", "WS-12.8"], "R-10"),
}

tasks = []
for tid, d in dag.items():
    if d.get("wave") != 1:
        continue
    r = rows.get(tid, {})
    extra, rule = EXTRA.get(tid, ([], ""))
    deps = sorted(set(d.get("deps_same_phase", [])) | set(r.get("deps", [])) | set(extra))
    tasks.append({
        "id": tid, "phase": d.get("phase") or "B", "title": r.get("title", ""),
        "review": r.get("review", "standard"), "model": r.get("model", "sonnet"),
        "real_infra": bool(d.get("real_infra")), "height": d.get("height", 0),
        "mutex": d.get("mutex", []), "deps": deps, "extra_deps_rule": rule,
        "status": r.get("status", "todo"),
    })
# WS-12.F1 was added by a controller Ruling (2026-11-03) after the DAG was built; schedule it on the
# schema lane right after WS-12.4 (same slot file 014), before phase B touches the safety repositories.
if "WS-12.F1" in rows and "WS-12.F1" not in dag:
    r = rows["WS-12.F1"]
    m124 = dag["WS-12.4"]["mutex"]
    tasks.append({
        "id": "WS-12.F1", "phase": "A", "title": r["title"], "review": r["review"], "model": r["model"],
        "real_infra": True, "height": 1,
        "mutex": sorted(set([k for k in m124 if "migration" in k]) | {
            "migrations/versions/014_safety.py", "tests/migrations/test_chain_real_pg.py",
            "suite:tests/migrations", "suite:services/safety_service/tests",
            "tests/realinfra/safety/test_mvp_041_safety_rls.py"}),
        "deps": sorted(set(r["deps"]) | {"WS-12.4"}), "extra_deps_rule": "plan (controller Ruling 2026-11-03)",
        "status": r["status"],
    })
missing = [t for t in rows if t not in dag and t != "WS-12.F1" and rows[t]["status"] != "done"]
tasks.sort(key=lambda t: (t["phase"], -t["height"], t["id"]))
OUT.write_text(json.dumps(tasks, indent=1), encoding="utf-8")
ph = {}
for t in tasks:
    ph.setdefault(t["phase"], []).append(t)
for p, ts in sorted(ph.items()):
    print(p, len(ts), "tasks;", sum(t["status"] == "done" for t in ts), "done;",
          "models", {m: sum(t["model"] == m for t in ts) for m in {t["model"] for t in ts}},
          "review", {m: sum(t["review"] == m for t in ts) for m in {t["review"] for t in ts}},
          "RI", sum(t["real_infra"] for t in ts))
print("tracker rows not in DAG and not done:", missing[:20])
print("WS-12.4 mutex:", dag["WS-12.4"]["mutex"])
ids = {t["id"] for t in tasks}
for t in tasks:
    bad = [d for d in t["deps"] if d not in ids and rows.get(d, {}).get("status") != "done"]
    if bad:
        print("dep outside W1 and not done:", t["id"], bad)
print("migrations-key holders:", [t["id"] for t in tasks if any("migration" in k for k in t["mutex"])])
