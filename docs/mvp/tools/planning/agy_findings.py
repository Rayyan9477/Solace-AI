"""Collect AGY grounding-review findings and append them to D:\\tmp\\mvp-plan\\fix\\<WS>.md.

usage: agy_findings.py <jobId> [<jobId> ...]
Reads the plugin's job records (result.rawOutput), extracts the last fenced json array, validates task ids
against task_index.json, writes agy_findings.json and appends a section per workstream to fix/<WS>.md.
"""
import json, os, re, sys, glob, collections

P = r"D:\tmp\mvp-plan"
STATE = r"C:\Users\rayyan.a\.claude\plugins\data\gemini-agy-plugin-cc\state"
tasks = json.load(open(os.path.join(P, "task_index.json"), encoding="utf-8"))
# AGY_OUT=<name> writes findings to <name>.json and fix-agy\<WS>.md instead of agy_findings.json / fix\<WS>.md
OUTNAME = os.environ.get("AGY_OUT", "agy_findings")
FIXDIR = "fix" if OUTNAME == "agy_findings" else "fix-agy"

all_f, report = [], []
for jid in sys.argv[1:]:
    hits = glob.glob(os.path.join(STATE, "*", "jobs", f"{jid}.json"))
    if not hits:
        report.append(f"{jid}: job record not found")
        continue
    rec = json.load(open(hits[0], encoding="utf-8"))
    res = rec.get("result") or {}
    raw = res.get("rawOutput") or res.get("output") or ""
    status = rec.get("status") or res.get("status")
    blocks = re.findall(r"```json\s*(.*?)```", raw, re.S)
    items, tail = None, ""
    for b in reversed(blocks):
        try:
            items, _ = json.JSONDecoder().raw_decode(b.strip())
            break
        except Exception:
            continue
    if items is None:
        m = raw.rfind("[")
        report.append(f"{jid}: status={status} NO PARSEABLE JSON (raw {len(raw)} chars)")
        continue
    m = re.search(r"checked:\s*(\d+)\s*tasks in\s*(\d+)\s*plans", raw)
    report.append(f"{jid}: status={status} findings={len(items)} " + (f"checked {m.group(1)} tasks in {m.group(2)} plans" if m else "no 'checked:' line"))
    for it in items:
        it["job"] = jid
        all_f.append(it)

bad = [f for f in all_f if f.get("task") not in tasks]
for f in all_f:
    if f.get("task") in tasks:
        f["ws"] = tasks[f["task"]]["ws"]
json.dump(all_f, open(os.path.join(P, OUTNAME + ".json"), "w", encoding="utf-8"), indent=1)

by_ws = collections.defaultdict(list)
for f in all_f:
    by_ws[f.get("ws", "?")].append(f)
os.makedirs(os.path.join(P, FIXDIR), exist_ok=True)
for ws, lst in sorted(by_ws.items()):
    if not re.match(r"WS-\d\d$", ws):
        continue
    path = os.path.join(P, FIXDIR, f"{ws}.md")
    head = "" if os.path.exists(path) else f"# Plan-check findings for {ws}\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(head + "\n## AGY grounding review (independent model; verify against the code before accepting)\n")
        for f in sorted(lst, key=lambda x: (x.get("severity") != "error", x.get("task", ""))):
            fh.write(f"- [{f.get('severity')}/{f.get('check')}] {f.get('task')}: claim: {f.get('claim')} | evidence: {f.get('evidence')} | fix: {f.get('fix')}\n")

for r in report:
    print(r)
print("total findings:", len(all_f), "| unknown task ids:", len(bad), [b.get("task") for b in bad][:15])
print("by ws:", {k: len(v) for k, v in sorted(by_ws.items())})
print("by check:", dict(collections.Counter(f.get("check") for f in all_f)))
print("by severity:", dict(collections.Counter(f.get("severity") for f in all_f)))
