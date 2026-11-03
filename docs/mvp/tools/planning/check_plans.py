"""Check the detailed workstream plans and derive the execution schedule.

usage: check_plans.py <detail-or-fix workflow output.json> [more outputs...]   (later outputs override earlier per ws)
writes D:\\tmp\\mvp-plan\\{task_index.json, plan_tasks.json, schedule.json, check_report.md, fix\\<WS>.md}
"""
import json, os, sys, re, collections

P = r"D:\tmp\mvp-plan"
REPO = r"D:\Repo\Contextual-Chatbot"
skel = json.load(open(os.path.join(P, "skeleton.json"), encoding="utf-8"))
reg = json.load(open(os.path.join(P, "register_compiled.json"), encoding="utf-8"))
required = [e["mid"] for e in reg if e["sev"] != "post-mvp"]
sev_of = {e["mid"]: e["sev"] for e in reg}

indexes = {}
for path in sys.argv[1:]:
    wf = json.load(open(path, encoding="utf-8"))
    res = wf.get("result", wf)
    for ix in res.get("indexes", []):
        indexes[ix["ws"]] = ix

ws_order = [w["id"] for w in skel["workstreams"]]
issues = collections.defaultdict(list)
tasks = {}
for ws in ws_order:
    ix = indexes.get(ws)
    if not ix:
        issues[ws].append("No plan/index returned for this workstream.")
        continue
    fpath = os.path.join(REPO, ix["file"].replace("/", os.sep))
    text = open(fpath, encoding="utf-8").read() if os.path.exists(fpath) else ""
    if not text:
        issues[ws].append(f"Plan file missing: {ix['file']}")
    for t in ix["tasks"]:
        t["ws"] = ws
        if t["id"] in tasks:
            issues[ws].append(f"Duplicate task id {t['id']}")
        tasks[t["id"]] = t
        if text and f"### Task {t['id']}:" not in text:
            issues[ws].append(f"Task {t['id']} is in the index but has no '### Task {t['id']}:' heading in the file")
    if text:
        for m in re.finditer(r"^### Task (WS-\d+\.\d+):", text, re.M):
            if m.group(1) not in {t["id"] for t in ix["tasks"]}:
                issues[ws].append(f"Task {m.group(1)} is in the file but missing from the index")
        for bad in ("TBD", "handle edge cases", "add appropriate validation", "write tests for the above"):
            if bad.lower() in text.lower():
                issues[ws].append(f"Placeholder phrase found in plan: '{bad}'")

# coverage
covered = collections.defaultdict(list)
for t in tasks.values():
    for m in t["covers"]:
        covered[m].append(t["id"])
assigned_ws = {m: w["id"] for w in skel["workstreams"] for m in w["issue_ids"]}
for m in required:
    if not covered.get(m):
        issues[assigned_ws.get(m, ws_order[0])].append(f"Register entry {m} ({sev_of[m]}) is not covered by any task")
for m in covered:
    if m not in sev_of:
        issues[tasks[covered[m][0]]["ws"]].append(f"Task(s) {covered[m]} cover unknown id {m}")

# dependencies: existence, waves, cycles
for t in tasks.values():
    for d in t["depends_on"]:
        if d not in tasks:
            issues[t["ws"]].append(f"{t['id']} depends on unknown task {d}")
        elif tasks[d]["wave"] > t["wave"]:
            issues[t["ws"]].append(f"{t['id']} (wave {t['wave']}) depends on {d} in a LATER wave ({tasks[d]['wave']})")
color = {}
def dfs(u, stack):
    color[u] = 1
    for v in tasks[u]["depends_on"]:
        if v not in tasks:
            continue
        if color.get(v) == 1:
            issues[tasks[u]["ws"]].append(f"Dependency cycle: {' -> '.join(stack + [u, v])}")
        elif not color.get(v):
            dfs(v, stack + [u])
    color[u] = 2
for u in tasks:
    if not color.get(u):
        dfs(u, [])

# file conflicts across workstreams within a wave
def files_of(t):
    return {f.replace("\\", "/").split(" ")[0].strip("`") for f in t["files_create"] + t["files_modify"]}
by_wave = collections.defaultdict(list)
for t in tasks.values():
    by_wave[t["wave"]].append(t)
for wv, ts in by_wave.items():
    owner = collections.defaultdict(set)
    for t in ts:
        for f in files_of(t):
            owner[f].add(t["ws"])
    for f, wss in owner.items():
        if len(wss) > 1:
            for w in wss:
                issues[w].append(f"Wave {wv}: file `{f}` is modified by several workstreams in the same wave ({', '.join(sorted(wss))}); only one may own it in a wave")

# cross-workstream requests must be planned by the owner
for ws, ix in indexes.items():
    for r in ix.get("cross_ws_requests", []):
        owner = r["owner_ws"]
        if owner not in indexes:
            issues[ws].append(f"Cross-workstream request to unknown owner {owner}: {r['change'][:120]}")
            continue
        owner_files = set().union(*[files_of(t) for t in indexes[owner]["tasks"]]) if indexes[owner]["tasks"] else set()
        if not any(f.replace("\\", "/") in owner_files for f in r["files"]):
            issues[owner].append(f"Plan a task for the request from {ws} (needed by {r['needed_by_task']}): {r['change']} [files: {', '.join(r['files'])}]")

# lanes: union of tasks sharing a file or a dependency edge inside a wave
schedule = {}
for wv, ts in sorted(by_wave.items()):
    parent = {t["id"]: t["id"] for t in ts}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    first = {}
    for t in ts:
        for f in files_of(t):
            if f in first:
                parent[find(t["id"])] = find(first[f])
            else:
                first[f] = t["id"]
        for d in t["depends_on"]:
            if d in parent:
                parent[find(t["id"])] = find(d)
    lanes = collections.defaultdict(list)
    for t in ts:
        lanes[find(t["id"])].append(t["id"])
    def key(tid):
        a, b = tid.split(".")
        return (ws_order.index(tasks[tid]["ws"]) if tasks[tid]["ws"] in ws_order else 99, int(b))
    schedule[wv] = sorted([sorted(v, key=key) for v in lanes.values()], key=lambda v: key(v[0]))

plan_tasks = {m: sorted(set(v)) for m, v in covered.items()}
json.dump(tasks, open(os.path.join(P, "task_index.json"), "w", encoding="utf-8"), indent=1)
json.dump(plan_tasks, open(os.path.join(P, "plan_tasks.json"), "w", encoding="utf-8"), indent=1)
json.dump({str(k): v for k, v in schedule.items()}, open(os.path.join(P, "schedule.json"), "w", encoding="utf-8"), indent=1)

os.makedirs(os.path.join(P, "fix"), exist_ok=True)
for f in os.listdir(os.path.join(P, "fix")):
    os.remove(os.path.join(P, "fix", f))
for ws, lst in issues.items():
    if lst:
        with open(os.path.join(P, "fix", f"{ws}.md"), "w", encoding="utf-8") as fh:
            fh.write(f"# Automated plan-check findings for {ws}\n\n" + "\n".join(f"- {x}" for x in dict.fromkeys(lst)) + "\n")

n_issue = sum(len(set(v)) for v in issues.values())
models = collections.Counter(t["model"] for t in tasks.values())
reviews = collections.Counter(t["review"] for t in tasks.values())
with open(os.path.join(P, "check_report.md"), "w", encoding="utf-8") as fh:
    fh.write(f"tasks {len(tasks)}; workstreams with plans {len(indexes)}/{len(ws_order)}; findings {n_issue}\n")
    fh.write(f"coverage: {sum(1 for m in required if covered.get(m))}/{len(required)} required ids covered\n")
    fh.write(f"models {dict(models)}; reviews {dict(reviews)}\n")
    for wv, lanes in sorted(schedule.items()):
        fh.write(f"wave {wv}: {sum(len(l) for l in lanes)} tasks in {len(lanes)} lanes; largest lane {max(len(l) for l in lanes)} tasks\n")
print(open(os.path.join(P, "check_report.md"), encoding="utf-8").read())
print("workstreams with findings:", {ws: len(set(v)) for ws, v in issues.items() if v})
