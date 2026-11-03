import json, os, re, collections, sys

P = r"D:\tmp\mvp-plan"
cat = collections.Counter()
examples = collections.defaultdict(list)
for fn in os.listdir(os.path.join(P, "fix")):
    txt = open(os.path.join(P, "fix", fn), encoding="utf-8").read()
    sec = txt.split("## Warnings")[0]
    for line in sec.splitlines():
        if not line.startswith("- ") or line == "- none":
            continue
        s = line[2:]
        if "No plan/index returned" in s:
            k = "no plan"
        elif "is not covered by any task" in s:
            k = "uncovered"
        elif s.startswith("Plan a task for the request"):
            k = "request not planned by owner"
        elif "which " in s and "owns (rule" in s:
            k = "edits file owned by another ws"
        elif "does not exist and is not created" in s:
            k = "modify of missing file"
        elif "depends on unknown task" in s:
            k = "unknown dep"
        elif "LATER wave" in s:
            k = "later-wave dep"
        elif "same wave" in s:
            k = "same-wave cross-ws file conflict"
        elif "is phase A" in s:
            k = "phase rule"
        elif "created by several tasks" in s:
            k = "created by several tasks"
        elif "Placeholder" in s:
            k = "placeholder"
        else:
            k = "other"
        cat[k] += 1
        if len(examples[k]) < 6:
            examples[k].append(f"[{fn[:-3]}] {s[:260]}")
for k, v in cat.most_common():
    print(f"{v:4}  {k}")
    for e in examples[k]:
        print("        ", e)

# lane analysis
tasks = json.load(open(os.path.join(P, "task_index.json"), encoding="utf-8"))
sched = json.load(open(os.path.join(P, "schedule.json"), encoding="utf-8"))
sys.path.insert(0, P)
src = open(os.path.join(P, "check_plans2.py"), encoding="utf-8").read()
ns = {"skel": json.load(open(os.path.join(P, "skeleton.json"), encoding="utf-8")), "os": os}
exec(src[src.index("def norm(f):"):src.index("def matches(")], ns)
exec(src[src.index("PLATFORM = ["):src.index("# ---------- per-workstream checks ----------")], {**ns, "fnmatch": __import__("fnmatch")}, ns)
for wv in ("0", "1"):
    for ph in ("A", "B"):
        lanes = sched.get(wv, {}).get(ph, [])
        if not lanes:
            continue
        big = max(lanes, key=len)
        keys = collections.defaultdict(set)
        for tid in big:
            t = tasks[tid]
            fs = {ns["norm"](f) for f in t["files_create"] + t["files_modify"]}
            for f in fs:
                keys[f].add(t["ws"])
            for f in fs | {ns["norm"](x) for x in t["files_test"]}:
                keys["suite:" + ns["suite_root"](f)].add(t["ws"])
        print(f"\nW{wv}{ph} biggest lane {len(big)} tasks, ws={sorted({tasks[t]['ws'] for t in big})}")
        print("   cross-ws linking keys:", sorted([(k, sorted(v)) for k, v in keys.items() if len(v) > 1], key=lambda x: -len(x[1]))[:14])
        deps = [(t, d) for t in big for d in tasks[t]["depends_on"] if d in big and tasks[d]["ws"] != tasks[t]["ws"]]
        print("   cross-ws same-phase deps:", deps[:14])
