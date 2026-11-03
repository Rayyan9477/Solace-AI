"""Check the detailed workstream plans and derive the execution schedule (v2).

usage: check_plans2.py [--lanes file|suite] <workflow output.json> [more outputs...]
  later outputs override earlier ones per workstream.
writes D:\\tmp\\mvp-plan\\{task_index.json, plan_tasks.json, schedule.json, check_report.md, fix\\<WS>.md}

checks: index/file consistency, placeholders, coverage, unknown/later-wave deps, cycles,
same-wave cross-workstream file conflicts, file ownership (most-specific owner wins; shared_files
protocol exceptions are warnings), path existence, phase rule (A before B), cross-workstream
requests planned by the owner. lanes: union-find per (wave, phase) on shared files and same-phase deps
(and, with --lanes suite, on the suite root a task's files belong to).
"""
import json, os, sys, re, collections, fnmatch

P = r"D:\tmp\mvp-plan"
REPO = r"D:\Repo\Contextual-Chatbot"
args = sys.argv[1:]
mode = "suite"
if args and args[0] == "--lanes":
    mode, args = args[1], args[2:]

skel = json.load(open(os.path.join(P, "skeleton.json"), encoding="utf-8"))
reg = json.load(open(os.path.join(P, "register_compiled.json"), encoding="utf-8"))
reg = reg["entries"] if isinstance(reg, dict) else reg
required = [e["mid"] for e in reg if e["sev"] != "post-mvp"]
sev_of = {e["mid"]: e["sev"] for e in reg}
ws_order = [w["id"] for w in skel["workstreams"]]
ws_by_id = {w["id"]: w for w in skel["workstreams"]}

indexes = {}
for path in args:
    wf = json.load(open(path, encoding="utf-8"))
    res = wf.get("result", wf)
    if isinstance(res, str):
        res = json.loads(res)
    for ix in res.get("indexes", []):
        indexes[ix["ws"]] = ix

errors = collections.defaultdict(list)
warns = collections.defaultdict(list)


def norm(f):
    f = f.replace("\\", "/").strip().strip("`").split(" (")[0].split(" ")[0].strip("`")
    return f[2:] if f.startswith("./") else f


# ---------- ownership resolver ----------
rules = []  # (pattern, owner, source, protocol)
for w in skel["workstreams"]:
    for p in w["owned_paths"]:
        rules.append((norm(p).rstrip("/"), w["id"], "owned", ""))
for s in skel.get("shared_files", []):
    rules.append((norm(s["path"]).rstrip("/"), s["owner_ws"], "shared", s.get("protocol", "")))


def matches(path, pat):
    if any(c in pat for c in "*?["):
        return fnmatch.fnmatchcase(path, pat) or fnmatch.fnmatchcase(path, pat + "/*")
    return path == pat or path.startswith(pat + "/")


def spec(pat):
    segs = pat.split("/")
    return (len(segs), -pat.count("*"), len(pat))


def owners(path):
    c = [(spec(p), o, src, proto, p) for (p, o, src, proto) in rules if matches(path, p)]
    c.sort(reverse=True)
    return c


def is_test_path(f):
    return "/tests/" in f or f.startswith("tests/") or os.path.basename(f).startswith("test_") or f.endswith("conftest.py")


PLATFORM = ["src/*", "services/shared/*", "migrations/*", "infrastructure/*", "deploy/*", "docker/*", ".github/*",
            "docker-compose*.yml", "Caddyfile", ".env.example", "Makefile", "pytest.ini", "pyproject.toml",
            "requirements*", ".dockerignore", ".pre-commit-config.yaml", "services/*/Dockerfile",
            "services/*/requirements.txt", "services/*/src/infrastructure/event_bridge.py",
            "tests/conftest.py", "tests/realinfra/conftest.py", "alembic.ini", "__init__.py"]


def is_platform(f):
    return any(fnmatch.fnmatchcase(f, p) for p in PLATFORM)


def suite_root(f):
    parts = f.split("/")
    if parts[0] == "services" and len(parts) > 2:
        return "/".join(parts[:2])
    if parts[0] == "tests" and len(parts) > 3 and parts[1] in ("realinfra", "integration", "e2e", "contracts"):
        return "/".join(parts[:3])
    if parts[0] in ("src", "tests") and len(parts) > 2:
        return "/".join(parts[:2])
    return parts[0]


# ---------- per-workstream checks ----------
tasks = {}
for ws in ws_order:
    ix = indexes.get(ws)
    if not ix:
        errors[ws].append("No plan/index returned for this workstream.")
        continue
    fpath = os.path.join(REPO, ix["file"].replace("/", os.sep))
    text = open(fpath, encoding="utf-8").read() if os.path.exists(fpath) else ""
    if not text:
        errors[ws].append(f"Plan file missing: {ix['file']}")
    ids_in_index = {t["id"] for t in ix["tasks"]}
    for t in ix["tasks"]:
        t["ws"] = ws
        if not t["id"].startswith(ws + "."):
            errors[ws].append(f"Task id {t['id']} does not start with {ws}.")
        if t["id"] in tasks:
            errors[ws].append(f"Duplicate task id {t['id']}")
        tasks[t["id"]] = t
        if text and f"### Task {t['id']}:" not in text:
            errors[ws].append(f"Task {t['id']} is in the index but has no '### Task {t['id']}:' heading in the file")
        if t["wave"] not in ws_by_id[ws]["waves"]:
            warns[ws].append(f"Task {t['id']} is in wave {t['wave']}, which is not one of {ws}'s skeleton waves {ws_by_id[ws]['waves']}")
    if text:
        for m in re.finditer(r"^### Task (WS-\d+\.\w+):", text, re.M):
            if m.group(1) not in ids_in_index:
                errors[ws].append(f"Task {m.group(1)} is in the file but missing from the index")
        for bad in ("TBD", "handle edge cases", "add appropriate validation", "write tests for the above"):
            if re.search(r"\b" + re.escape(bad) + r"\b", text, re.I):
                errors[ws].append(f"Placeholder phrase found in plan: '{bad}'")
    for m in ix.get("uncovered_ids", []):
        errors[ws].append(f"Planner reported {m} as uncovered")

# ---------- coverage ----------
covered = collections.defaultdict(list)
for t in tasks.values():
    for m in t["covers"]:
        covered[m].append(t["id"])
assigned_ws = {m: w["id"] for w in skel["workstreams"] for m in w["issue_ids"]}
for m in required:
    if not covered.get(m):
        errors[assigned_ws.get(m, ws_order[0])].append(f"Register entry {m} ({sev_of[m]}) is not covered by any task")
for m, tids in covered.items():
    if m not in sev_of:
        errors[tasks[tids[0]]["ws"]].append(f"Task(s) {tids} cover unknown id {m}")
    else:
        if assigned_ws.get(m) and all(tasks[t]["ws"] != assigned_ws[m] for t in tids):
            warns[tasks[tids[0]]["ws"]].append(f"{m} is assigned to {assigned_ws[m]} but covered only by {tids}")

# ---------- dependencies ----------
for t in tasks.values():
    raw = t.get("depends_raw", "")
    vague = [m for m in re.findall(r"\bWS-\d{2}\b(?!\.[A-Za-z0-9])[^,;]*", raw) if not m.startswith(t["ws"])]
    if vague:
        warns[t["ws"]].append(f"{t['id']} names a workstream or interface instead of a task in 'Depends on' ({'; '.join(v.strip()[:60] for v in vague)}); name the producing task id so the scheduler can order it")
    for d in t["depends_on"]:
        if d not in tasks:
            errors[t["ws"]].append(f"{t['id']} depends on unknown task {d}")
        elif tasks[d]["wave"] > t["wave"]:
            errors[t["ws"]].append(f"{t['id']} (wave {t['wave']}) depends on {d} in a LATER wave ({tasks[d]['wave']})")
color = {}


def dfs(u, stack):
    color[u] = 1
    for v in tasks[u]["depends_on"]:
        if v not in tasks:
            continue
        if color.get(v) == 1:
            errors[tasks[u]["ws"]].append(f"Dependency cycle: {' -> '.join(stack + [u, v])}")
        elif not color.get(v):
            dfs(v, stack + [u])
    color[u] = 2


for u in list(tasks):
    if not color.get(u):
        dfs(u, [])


def files_of(t):
    return {norm(f) for f in t["files_create"] + t["files_modify"] if norm(f)}


# ---------- phase ----------
# A: touches a platform path. B: touches a service path (and no platform path).
# Neutral tasks (tests/, scripts/, docs/ only) float: B if a same-wave dependency is B, else A.
def is_service(f):
    parts = f.split("/")
    return parts[0] == "services" and len(parts) > 2 and parts[1] != "shared" and not is_platform(f)


for t in tasks.values():
    fs = files_of(t) | {norm(x) for x in t["files_test"]}
    if any(is_platform(f) for f in files_of(t)):
        t["phase"] = "A"
    elif any(is_service(f) for f in fs):
        t["phase"] = "B"
    else:
        t["phase"] = None
_ph_memo = {}


def float_phase(tid, seen=()):
    t = tasks[tid]
    if t["phase"] is not None:
        return t["phase"]
    if tid in _ph_memo:
        return _ph_memo[tid]
    ph = "A"
    for d in t["depends_on"]:
        if d in tasks and tasks[d]["wave"] == t["wave"] and d not in seen and float_phase(d, seen + (tid,)) == "B":
            ph = "B"
    _ph_memo[tid] = ph
    return ph


for tid in tasks:
    if tasks[tid]["phase"] is None:
        tasks[tid]["phase_floated"] = True
for tid in tasks:
    tasks[tid]["phase"] = float_phase(tid)
for t in tasks.values():
    for d in t["depends_on"]:
        if d in tasks and tasks[d]["wave"] == t["wave"] and t["phase"] == "A" and tasks[d]["phase"] == "B":
            errors[t["ws"]].append(f"{t['id']} is phase A (platform files) but depends on same-wave phase-B task {d}; move one of them or split the task")

# ---------- ownership ----------
def own_test_dir(f, ws):
    m = re.match(r"tests/(realinfra|integration|e2e|contracts|clinical)/([^/]+)/", f)
    if not m:
        return False
    d = m.group(2).lower()
    slug = ws_by_id[ws]["slug"].lower()
    return d in (slug, ws.lower(), ws.lower().replace("-", "")) or slug.startswith(d) or d.startswith(slug.split("-")[0])


for t in tasks.values():
    for f in files_of(t):
        if own_test_dir(f, t["ws"]):
            continue
        cand = owners(f)
        if not cand:
            if not is_test_path(f):
                warns[t["ws"]].append(f"{t['id']} edits `{f}`, which no workstream owns in the skeleton")
            continue
        _, owner, src, proto, pat = cand[0]
        if owner == t["ws"]:
            continue
        sanctioned = t["ws"] in proto or any(o == t["ws"] and s == "shared" for (_, o, s, _, _) in cand)
        if is_test_path(f) or sanctioned:
            warns[t["ws"]].append(f"{t['id']} edits `{f}` owned by {owner} (rule `{pat}`); allowed only if the shared-file protocol sanctions it")
        else:
            errors[t["ws"]].append(f"{t['id']} edits `{f}`, which {owner} owns (rule `{pat}`); move it to a cross-workstream request")

# ---------- same-wave cross-workstream file conflicts ----------
_anc = {}


def anc_of(tid):
    if tid in _anc:
        return _anc[tid]
    _anc[tid] = set()
    out = set()
    for d in tasks[tid]["depends_on"]:
        if d in tasks:
            out.add(d)
            out |= anc_of(d)
    _anc[tid] = out
    return out


by_wave = collections.defaultdict(list)
for t in tasks.values():
    by_wave[t["wave"]].append(t)
for wv, ts in by_wave.items():
    owner = collections.defaultdict(list)
    for t in ts:
        for f in files_of(t):
            owner[f].append(t)
    for f, tl in owner.items():
        wss = sorted({t["ws"] for t in tl})
        if len(wss) > 1:
            ordered = all(a["id"] in anc_of(b["id"]) or b["id"] in anc_of(a["id"])
                          for i, a in enumerate(tl) for b in tl[i + 1:] if a["ws"] != b["ws"])
            for w in wss:
                (warns if ordered else errors)[w].append(
                    f"Wave {wv}: file `{f}` is edited by several workstreams in the same wave ({', '.join(t['id'] for t in tl)})"
                    + ("; they are ordered by dependencies, so check the hand-over is intended" if ordered else "; add a dependency or move one edit to the owner"))

# ---------- path existence ----------
created_by = collections.defaultdict(list)
for t in tasks.values():
    for f in t["files_create"]:
        created_by[norm(f)].append(t["id"])


def ancestors(tid, seen=None):
    seen = set() if seen is None else seen
    for d in tasks[tid]["depends_on"]:
        if d in tasks and d not in seen:
            seen.add(d)
            ancestors(d, seen)
    return seen


for t in tasks.values():
    anc = ancestors(t["id"])
    for f in t["files_modify"]:
        nf = norm(f)
        if not nf or any(c in nf for c in "*?"):
            continue
        if os.path.exists(os.path.join(REPO, nf.replace("/", os.sep))):
            continue
        makers = created_by.get(nf, [])
        ok = [m for m in makers if m in anc or tasks[m]["wave"] < t["wave"]]
        if not ok:
            msg = f"{t['id']} modifies `{nf}`, which does not exist and is not created by an earlier-wave task or a dependency"
            (errors if not makers else warns)[t["ws"]].append(msg + (f" (created by {makers})" if makers else ""))
    for f in t["files_create"]:
        nf = norm(f)
        if nf and os.path.exists(os.path.join(REPO, nf.replace("/", os.sep))):
            warns[t["ws"]].append(f"{t['id']} lists `{nf}` under Create, but it already exists (should be Modify?)")
    for f in t["files_create"]:
        nf = norm(f)
        if len(created_by.get(nf, [])) > 1:
            errors[t["ws"]].append(f"`{nf}` is created by several tasks: {created_by[nf]}")

# ---------- cross-workstream requests ----------
for ws, ix in indexes.items():
    for r in ix.get("cross_ws_requests", []):
        owner = r["owner_ws"]
        if owner not in indexes:
            errors[ws].append(f"Cross-workstream request to unknown owner {owner}: {r['change'][:160]}")
            continue
        owner_files = set().union(*[files_of(t) for t in indexes[owner]["tasks"]]) if indexes[owner]["tasks"] else set()
        if not r["files"]:
            warns[owner].append(f"Request from {ws} (needed by {r['needed_by_task']}) names no files; confirm a task covers it: {r['change'][:220]}")
        elif not any(norm(f) in owner_files for f in r["files"]):
            errors[owner].append(f"Plan a task for the request from {ws} (needed by {r['needed_by_task']}): {r['change']} [files: {', '.join(r['files'])}]")
        nb = r.get("needed_by_task", "")
        if nb and nb not in tasks:
            warns[ws].append(f"Cross-workstream request names unknown needing task {nb}")

# ---------- schedule: dependency-ready dispatch with mutual exclusion ----------
WORKERS, INFRA = 6, 2


def suite_key(f):
    parts = f.split("/")
    if parts[0] == "services" and len(parts) > 2:
        if parts[1] == "shared":
            return "services/shared"
        if parts[-1] in ("Dockerfile", "requirements.txt") and len(parts) == 3:
            return None
        return "services/" + parts[1]
    if parts[0] == "src" and len(parts) > 2:
        return "src/" + parts[1]
    if parts[0] == "tests":
        if len(parts) > 3 and parts[1] in ("realinfra", "integration", "e2e", "contracts", "clinical"):
            return "/".join(parts[:3])
        if len(parts) > 2:
            return "tests/" + parts[1]
        return None
    if parts[0] == "migrations":
        return "migrations"
    if parts[0] == "infrastructure" and len(parts) > 2:
        return "infrastructure/" + parts[1]
    return None


def mutex_keys(t):
    keys = set(files_of(t))
    for f in files_of(t) | {norm(x) for x in t["files_test"]}:
        k = suite_key(f)
        if k:
            keys.add("suite:" + k)
    return keys


def tkey(tid):
    ws, k = tid.split(".", 1)
    return (ws_order.index(ws) if ws in ws_order else 99, int(re.sub(r"\D", "", k) or 0), k)


schedule, dag = {}, {}
for wv, ts in sorted(by_wave.items()):
    schedule[wv] = {}
    for ph in ("A", "B"):
        group = [t for t in ts if t["phase"] == ph]
        if not group:
            continue
        ids = {t["id"] for t in group}
        deps = {t["id"]: [d for d in t["depends_on"] if d in ids] for t in group}
        children = collections.defaultdict(list)
        for x, ds in deps.items():
            for d in ds:
                children[d].append(x)
        hmemo = {}

        def height(x, stack=()):
            if x in hmemo:
                return hmemo[x]
            if x in stack:
                return 1
            hmemo[x] = 1 + max((height(c, stack + (x,)) for c in children[x]), default=0)
            return hmemo[x]

        lmemo = {}

        def level(x, stack=()):
            if x in lmemo:
                return lmemo[x]
            if x in stack:
                return 1
            lmemo[x] = 1 + max((level(d, stack + (x,)) for d in deps[x]), default=0)
            return lmemo[x]

        keys = {t["id"]: mutex_keys(t) for t in group}
        sev_rank = {"blocker": 0, "high": 1, "medium": 2, "low": 3}
        crit_rank = {"safety-critical": 0, "security-critical": 1, "standard": 2}
        done, remaining, sim = set(), set(ids), []
        while remaining:
            ready = sorted([x for x in remaining if all(d in done for d in deps[x])],
                           key=lambda x: (-height(x), sev_rank.get(tasks[x]["severity"], 9), crit_rank.get(tasks[x]["review"], 9), tkey(x)))
            batch, used, n_inf = [], set(), 0
            for x in ready:
                if len(batch) >= WORKERS:
                    break
                if keys[x] & used:
                    continue
                if tasks[x]["real_infra"] and n_inf >= INFRA:
                    continue
                batch.append(x)
                used |= keys[x]
                n_inf += 1 if tasks[x]["real_infra"] else 0
            if not batch:
                batch = [ready[0]] if ready else [sorted(remaining, key=tkey)[0]]
            for x in batch:
                done.add(x)
                remaining.discard(x)
            sim.append(sorted(batch, key=tkey))
        levels = collections.defaultdict(list)
        for x in ids:
            levels[level(x)].append(x)
        schedule[wv][ph] = {
            "tasks": len(ids),
            "critical_path": max((height(x) for x in ids), default=0),
            "rounds": len(sim),
            "levels": [sorted(levels[k], key=tkey) for k in sorted(levels)],
            "sim": sim,
        }
        for t in group:
            dag[t["id"]] = {"wave": wv, "phase": ph, "deps_same_phase": deps[t["id"]], "mutex": sorted(keys[t["id"]]),
                            "height": height(t["id"]), "real_infra": t["real_infra"]}
json.dump(dag, open(os.path.join(P, "schedule_dag.json"), "w", encoding="utf-8"), indent=1)

# ---------- outputs ----------
plan_tasks = {m: sorted(set(v)) for m, v in covered.items()}
json.dump(tasks, open(os.path.join(P, "task_index.json"), "w", encoding="utf-8"), indent=1)
json.dump(indexes, open(os.path.join(P, "indexes.json"), "w", encoding="utf-8"), indent=1)
json.dump(plan_tasks, open(os.path.join(P, "plan_tasks.json"), "w", encoding="utf-8"), indent=1)
json.dump({str(k): v for k, v in schedule.items()}, open(os.path.join(P, "schedule.json"), "w", encoding="utf-8"), indent=1)

os.makedirs(os.path.join(P, "fix"), exist_ok=True)
for f in os.listdir(os.path.join(P, "fix")):
    os.remove(os.path.join(P, "fix", f))
for ws in ws_order:
    e, w = list(dict.fromkeys(errors.get(ws, []))), list(dict.fromkeys(warns.get(ws, [])))
    if e or w:
        with open(os.path.join(P, "fix", f"{ws}.md"), "w", encoding="utf-8") as fh:
            fh.write(f"# Automated plan-check findings for {ws}\n\n## Errors (must fix)\n")
            fh.write("\n".join(f"- {x}" for x in e) or "- none")
            fh.write("\n\n## Warnings (check; fix or justify in the plan's Self-review)\n")
            fh.write("\n".join(f"- {x}" for x in w) or "- none")
            fh.write("\n")

n_err = sum(len(set(v)) for v in errors.values())
n_warn = sum(len(set(v)) for v in warns.values())
models = collections.Counter(t["model"] for t in tasks.values())
reviews = collections.Counter(t["review"] for t in tasks.values())
with open(os.path.join(P, "check_report.md"), "w", encoding="utf-8") as fh:
    fh.write(f"lane mode {mode}; tasks {len(tasks)}; workstreams with plans {len(indexes)}/{len(ws_order)}; errors {n_err}; warnings {n_warn}\n")
    fh.write(f"coverage: {sum(1 for m in required if covered.get(m))}/{len(required)} required ids covered\n")
    fh.write(f"models {dict(models)}; reviews {dict(reviews)}; real-infra tasks {sum(1 for t in tasks.values() if t['real_infra'])}\n")
    tot = 0
    for wv, phs in sorted(schedule.items()):
        for ph, s in phs.items():
            tot += s["rounds"]
            fh.write(f"wave {wv} phase {ph}: {s['tasks']} tasks; critical path {s['critical_path']}; "
                     f"simulated rounds ({WORKERS} workers, {INFRA} real-infra) {s['rounds']}; max parallel {max(len(b) for b in s['sim'])}\n")
    fh.write(f"total simulated rounds: {tot} for {len(tasks)} tasks\n")
print(open(os.path.join(P, "check_report.md"), encoding="utf-8").read())
print("errors per ws:", {ws: len(set(v)) for ws, v in errors.items() if v})
print("warnings per ws:", {ws: len(set(v)) for ws, v in warns.items() if v})
