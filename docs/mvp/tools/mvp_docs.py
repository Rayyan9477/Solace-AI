"""Maintain the MVP tracking docs in docs/mvp/ (standard library only).

Commands (run from anywhere; paths are resolved from this file's location):
  build [--schedule PATH]   Rebuild TRACKER.md task tables from the workstream plans. Keeps every
                            task's Status and Notes cells, adds new tasks as `todo` and lists tasks
                            that vanished from the plans. Phase and dependency level come from the
                            scheduler output (default D:\\tmp\\mvp-plan\\schedule_dag.json); a trimmed
                            copy is kept in docs/mvp/data/tasks.json so later builds work without it.
  set TASK STATUS [--note TEXT]
                            Set one task's Status (and optionally its Notes), then refresh the counts.
  refresh                   Recount the summaries in TRACKER.md and README.md without reading the plans.
  check                     Report invalid statuses and tasks missing from the tracker or the plans.

Statuses: todo, in-progress, review, fixing, done, blocked, deferred.
"""
import argparse, json, os, re, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
MVP = os.path.dirname(HERE)                       # docs/mvp
REPO = os.path.dirname(os.path.dirname(MVP))      # repo root
PLAN_DIR = os.path.join(REPO, "docs", "superpowers", "plans", "2026-10-18-mvp-remediation")
PLAN_REL = "../superpowers/plans/2026-10-18-mvp-remediation"
TRACKER = os.path.join(MVP, "TRACKER.md")
README = os.path.join(MVP, "README.md")
DATA = os.path.join(MVP, "data", "tasks.json")
DEFAULT_SCHEDULE = r"D:\tmp\mvp-plan\schedule_dag.json"
STATUSES = ["todo", "in-progress", "review", "fixing", "done", "blocked", "deferred"]
ACTIVE = {"in-progress", "review", "fixing"}
TID = r"WS-\d{2}\.[A-Za-z0-9]+"
WAVE_NAMES = {0: "Build, migrate and boot", 1: "Crisis path", 2: "Chat turn and durable state",
              3: "Privacy controls", 4: "Enterprise basics", 5: "Residual defects and release gate"}
SUM_START, SUM_END = "<!-- mvp:summary:start -->", "<!-- mvp:summary:end -->"
ROW_RE = re.compile(r"^\|\s*\[?(" + TID + r")\]?")


def slug(text):
    s = text.strip().lower()
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def field(block, name):
    m = re.search(r"^\*\*" + re.escape(name) + r":\*\*\s*(.*)$", block, re.M)
    return m.group(1).strip() if m else ""


def parse_plans():
    tasks = collections.OrderedDict()
    for fn in sorted(os.listdir(PLAN_DIR)):
        m = re.match(r"(WS-\d{2})-.*\.md$", fn)
        if not m:
            continue
        text = open(os.path.join(PLAN_DIR, fn), encoding="utf-8").read()
        heads = list(re.finditer(r"^### Task (" + TID + r"):\s*(.*)$", text, re.M))
        for h in heads:
            start = h.end()
            nxt = re.search(r"^(### Task |## )", text[start:], re.M)
            block = text[start:start + nxt.start()] if nxt else text[start:]
            tid, title = h.group(1), h.group(2).strip()
            sev = field(block, "Severity").lower()
            rv = field(block, "Review").lower()
            mdl = field(block, "Implementer model").lower()
            wv = re.search(r"\d+", field(block, "Wave"))
            tasks[tid] = {
                "id": tid, "ws": m.group(1), "title": title, "file": fn,
                "anchor": slug(f"Task {tid}: {title}"),
                "wave": int(wv.group()) if wv else -1,
                "severity": next((s for s in ("blocker", "high", "medium", "low") if s in sev), "?"),
                "review": next((r for r in ("safety-critical", "security-critical", "standard") if r in rv), "?"),
                "model": "Opus" if "opus" in mdl else ("Sonnet" if "sonnet" in mdl else "?"),
                "real_infra": field(block, "Real infra").lower().startswith("yes"),
                "covers": sorted(set(re.findall(r"MVP-\d{3}", field(block, "Covers")))),
                "depends": sorted(set(re.findall(TID, field(block, "Depends on"))) - {tid}),
            }
    return tasks


def read_rows():
    """Existing Status/Notes per task id from TRACKER.md."""
    rows = {}
    if not os.path.exists(TRACKER):
        return rows
    for line in open(TRACKER, encoding="utf-8"):
        m = ROW_RE.match(line)
        if not m:
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
        if len(cells) >= 2:
            rows[m.group(1)] = {"status": cells[-2].strip("`"), "notes": cells[-1], "cells": cells}
    return rows


def load_meta(schedule):
    meta = {}
    if os.path.exists(DATA):
        meta = json.load(open(DATA, encoding="utf-8"))
    path = schedule or DEFAULT_SCHEDULE
    if os.path.exists(path):
        dag = json.load(open(path, encoding="utf-8"))
        meta = {k: {"phase": v.get("phase", "?"), "height": v.get("height", 0),
                    "deps_same_phase": v.get("deps_same_phase", [])} for k, v in dag.items()}
        os.makedirs(os.path.dirname(DATA), exist_ok=True)
        json.dump(meta, open(DATA, "w", encoding="utf-8"), indent=1, sort_keys=True)
    return meta


def esc(s):
    return s.replace("|", "\\|")


def summary(tasks, status):
    by = collections.Counter(status.get(t, "todo") for t in tasks)
    L = [SUM_START, "", f"*Counts refreshed by `mvp_docs.py`; edit Status cells, not these numbers.*", ""]
    L.append("| Status | " + " | ".join(STATUSES) + " | total |")
    L.append("|---|" + "---|" * (len(STATUSES) + 1))
    L.append("| Tasks | " + " | ".join(str(by.get(s, 0)) for s in STATUSES) + f" | {len(tasks)} |")
    L.append("")
    L.append("| Wave | Phase A done / total | Phase B done / total | Active | Blocked |")
    L.append("|---|---|---|---|---|")
    for wv in sorted({t["wave"] for t in tasks.values()}):
        row = []
        for ph in ("A", "B"):
            ids = [k for k, t in tasks.items() if t["wave"] == wv and t.get("phase") == ph]
            row.append(f"{sum(1 for k in ids if status.get(k) == 'done')} / {len(ids)}")
        ws_ids = [k for k, t in tasks.items() if t["wave"] == wv]
        act = sum(1 for k in ws_ids if status.get(k) in ACTIVE)
        blk = sum(1 for k in ws_ids if status.get(k) == "blocked")
        L.append(f"| W{wv} | {row[0]} | {row[1]} | {act} | {blk} |")
    L.append("")
    L.append("| Workstream | Done / total |")
    L.append("|---|---|")
    for ws in sorted({t["ws"] for t in tasks.values()}):
        ids = [k for k, t in tasks.items() if t["ws"] == ws]
        L.append(f"| {ws} | {sum(1 for k in ids if status.get(k) == 'done')} / {len(ids)} |")
    cov = collections.defaultdict(list)
    for k, t in tasks.items():
        for m in t["covers"]:
            cov[m].append(k)
    fixed = sum(1 for m, ks in cov.items() if all(status.get(k) == "done" for k in ks))
    part = sum(1 for m, ks in cov.items() if any(status.get(k) == "done" for k in ks) and not all(status.get(k) == "done" for k in ks))
    L.append("")
    L.append(f"**Register entries:** {fixed} of {len(cov)} covered entries fixed (all their tasks done); {part} partly done.")
    L += ["", SUM_END]
    return "\n".join(L), {"by": by, "fixed": fixed, "entries": len(cov), "total": len(tasks)}


def replace_block(text, block):
    if SUM_START in text and SUM_END in text:
        a, b = text.index(SUM_START), text.index(SUM_END) + len(SUM_END)
        return text[:a] + block + text[b:]
    return text


def render(tasks, rows, meta):
    status = {k: (rows[k]["status"] if k in rows else "todo") for k in tasks}
    notes = {k: (rows[k]["notes"] if k in rows else "") for k in tasks}
    for k, t in tasks.items():
        mt = meta.get(k, {})
        t["phase"] = mt.get("phase", "?")
        t["height"] = mt.get("height", 0)
        t["deps_same_phase"] = mt.get("deps_same_phase", [])
    old = open(TRACKER, encoding="utf-8").read() if os.path.exists(TRACKER) else ""
    head, barrier = HEADER, BARRIER_DEFAULT
    if old:
        if "## Barrier log" in old:
            seg = old[old.index("## Barrier log"):]
            nxt = re.search(r"^## (?!Barrier log)", seg, re.M)
            barrier = seg[:nxt.start()] if nxt else seg
        if SUM_START in old:
            head = old[:old.index(SUM_START)]
    block, _ = summary(tasks, status)
    L = [head.rstrip() + "\n", block, "", barrier.rstrip() + "\n"]
    for wv in sorted({t["wave"] for t in tasks.values()}):
        L.append(f"## W{wv}: {WAVE_NAMES.get(wv, '')}\n")
        for ph in ("A", "B", "?"):
            ids = [k for k, t in tasks.items() if t["wave"] == wv and t["phase"] == ph]
            if not ids:
                continue
            ids.sort(key=lambda k: (-tasks[k]["height"], k))
            label = {"A": "Phase A (platform)", "B": "Phase B (services)", "?": "Phase not scheduled yet"}[ph]
            L.append(f"### W{wv} {label}: {len(ids)} tasks\n")
            L.append("| Task | Title | Sev | Review | Model | Infra | Waits on (same phase) | Covers | Status | Notes |")
            L.append("|---|---|---|---|---|---|---|---|---|---|")
            for k in ids:
                t = tasks[k]
                link = f"[{k}]({PLAN_REL}/{t['file']}#{t['anchor']})"
                deps = ", ".join(t["deps_same_phase"]) or "-"
                covers = ", ".join(t["covers"]) or "support"
                L.append(f"| {link} | {esc(t['title'])} | {t['severity']} | {t['review']} | {t['model']} | "
                         f"{'yes' if t['real_infra'] else 'no'} | {deps} | {covers} | `{status[k]}` | {notes[k]} |")
            L.append("")
    gone = sorted(set(rows) - set(tasks))
    if gone:
        L.append("## Tasks no longer in the plans\n")
        L.append("These rows were in the tracker but their task blocks are gone. Keep them until the ledger explains why.\n")
        L.append("| Task | Status | Notes |")
        L.append("|---|---|---|")
        for k in gone:
            L.append(f"| {k} | `{rows[k]['status']}` | {rows[k]['notes']} |")
        L.append("")
    open(TRACKER, "w", encoding="utf-8", newline="\n").write("\n".join(L))
    return status


def refresh_readme(tasks, status):
    if not os.path.exists(README):
        return
    _, s = summary(tasks, status)
    by = s["by"]
    block = "\n".join([
        SUM_START, "",
        f"- Tasks done: **{by.get('done', 0)} / {s['total']}** (active {sum(by.get(x, 0) for x in ACTIVE)}, blocked {by.get('blocked', 0)}, deferred {by.get('deferred', 0)}).",
        f"- Register entries fixed: **{s['fixed']} / {s['entries']}**.",
        "- Per-wave and per-workstream counts: [TRACKER.md](TRACKER.md).", "", SUM_END])
    text = open(README, encoding="utf-8").read()
    open(README, "w", encoding="utf-8", newline="\n").write(replace_block(text, block))


def tasks_from_tracker():
    """Rebuild a minimal task map from TRACKER.md alone (for refresh without plans)."""
    tasks = parse_plans()
    meta = load_meta(None)
    for k, t in tasks.items():
        t["phase"] = meta.get(k, {}).get("phase", "?")
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--schedule")
    st = sub.add_parser("set"); st.add_argument("task"); st.add_argument("status", choices=STATUSES); st.add_argument("--note")
    sub.add_parser("refresh"); sub.add_parser("check")
    a = ap.parse_args()
    if a.cmd == "build":
        tasks = parse_plans()
        status = render(tasks, read_rows(), load_meta(a.schedule))
        refresh_readme(tasks, status)
        print(f"tracker: {len(tasks)} tasks; " + ", ".join(f"{k} {v}" for k, v in collections.Counter(status.values()).items()))
    elif a.cmd == "set":
        rows = read_rows()
        if a.task not in rows:
            sys.exit(f"{a.task} is not in TRACKER.md (run build first)")
        lines = open(TRACKER, encoding="utf-8").read().split("\n")
        for i, line in enumerate(lines):
            m = ROW_RE.match(line)
            if m and m.group(1) == a.task:
                cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
                cells[-2] = f"`{a.status}`"
                if a.note is not None:
                    cells[-1] = esc(a.note)
                lines[i] = "| " + " | ".join(cells) + " |"
        open(TRACKER, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
        tasks = tasks_from_tracker()
        status = {k: v["status"] for k, v in read_rows().items()}
        text = open(TRACKER, encoding="utf-8").read()
        open(TRACKER, "w", encoding="utf-8", newline="\n").write(replace_block(text, summary(tasks, status)[0]))
        refresh_readme(tasks, status)
        print(f"{a.task} -> {a.status}")
    elif a.cmd == "refresh":
        tasks = tasks_from_tracker()
        status = {k: v["status"] for k, v in read_rows().items()}
        text = open(TRACKER, encoding="utf-8").read()
        open(TRACKER, "w", encoding="utf-8", newline="\n").write(replace_block(text, summary(tasks, status)[0]))
        refresh_readme(tasks, status)
        print("refreshed")
    elif a.cmd == "check":
        rows, tasks = read_rows(), parse_plans()
        bad = {k: v["status"] for k, v in rows.items() if v["status"] not in STATUSES}
        print("invalid statuses:", bad or "none")
        print("in plans, not in tracker:", sorted(set(tasks) - set(rows)) or "none")
        print("in tracker, not in plans:", sorted(set(rows) - set(tasks)) or "none")
        sys.exit(1 if bad else 0)


HEADER = """# MVP execution tracker

One row per plan task. **Status** is the only cell to edit by hand (or run
`python docs/mvp/tools/mvp_docs.py set WS-NN.K <status> --note "<tree id or reason>"`).
Allowed values: `todo`, `in-progress`, `review`, `fixing`, `done`, `blocked`, `deferred`.
A task is `done` only after its review is clean and its ledger line is written (master plan, section 5.2).
After editing, run `python docs/mvp/tools/mvp_docs.py refresh`; after the plans change, run `build`.
The register's statuses are derived from this file at each barrier (`fixed` when every covering task is done).

"""

BARRIER_DEFAULT = """## Barrier log

Filled at each wave barrier (master plan, section 5.5). The tree zip is the only durable copy of the barrier, because there are no commits (D-17).

| Wave | Date | Barrier tree | Zip copied off-machine | Suites green / test count | Register delta | Notes |
|---|---|---|---|---|---|---|
| W0 entry | | | | baseline (venv): | | |
| W0 | | | | | | |
| W1 | | | | | | |
| W2 | | | | | | |
| W3 | | | | | | |
| W4 | | | | | | |
| W5 | | | | | | |

"""

if __name__ == "__main__":
    main()
