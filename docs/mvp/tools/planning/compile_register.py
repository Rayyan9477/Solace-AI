"""Compile the MVP issue register.

Inputs (D:\\tmp\\mvp-review\\state and D:\\tmp\\mvp-plan):
  register.json, synth.json                 verified review (344 entries, clusters, possible overlaps)
  agy_overlaps.json                         AGY merge decisions for the 40 overlap groups
  regrade.json                              Opus re-grade under the confirmed scope (+ new entries)
  agy_crossref.json                         AGY mapping of earlier tracker items -> register ids
  gaps_verified.json (optional)             verified entries for tracker items no register entry covered
  plan_tasks.json (optional)                {mvp_id: [task ids]} produced by the planning step
Outputs:
  D:\\tmp\\mvp-plan\\register_compiled.json and docs/MVP-ISSUE-REGISTER.md
"""
import json, os, re, collections

ST = r"D:\tmp\mvp-review\state"
P = r"D:\tmp\mvp-plan"
DOC = os.environ.get("REG_DOC", r"D:\Repo\Contextual-Chatbot\docs\MVP-ISSUE-REGISTER.md")
SEVS = ["blocker", "high", "medium", "low", "post-mvp", "none"]
RANK = {s: 5 - i for i, s in enumerate(SEVS)}
SCOPE = ("**Pilot + enterprise basics** (confirmed by the owner on 2026-10-18). A single-region pilot with real users on one "
         "docker-compose host: email/password and Google sign-in, chat through the orchestrator REST/WebSocket API, every "
         "message safety-screened, a detected crisis shows resources and alerts an on-call clinician, durable PHI that is "
         "encrypted and isolated per user, working account deletion, health checks and basic monitoring/alerting. "
         "Plus four enterprise basics: backups with a tested restore, MFA, a working audit trail, and per-environment "
         "secrets management. Out of scope: HA/DR across zones, horizontal scaling, load/chaos testing, SOC2 certification "
         "evidence, the Kong gateway package, ClickHouse analytics, config_service as a product feature, and an enterprise "
         "RBAC/ABAC engine.")


def load(path, default=None):
    if not os.path.exists(path):
        return default
    return json.load(open(path, encoding="utf-8"))


def norm(s):
    return s if s in RANK else "medium"


def cell(s):
    return " ".join(str(s or "").split()).replace("|", "\\|")


def clip(s, n):
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


reg = load(os.path.join(ST, "register.json"))
syn = load(os.path.join(ST, "synth.json"))
over = load(os.path.join(P, "agy_overlaps.json"), [])
regr = load(os.path.join(P, "regrade.json"), {"regrades": [], "new_entries": [], "control_coverage": []})
xref = load(os.path.join(P, "agy_crossref.json"), [])
gaps = load(os.path.join(P, "gaps_verified.json"), {"entries": [], "rejected": []})
plan_tasks = load(os.path.join(P, "plan_tasks.json"), {})
status_over = load(os.path.join(P, "status.json"), {})  # execution: {mid: "fixed (WS-NN.K)" | "in progress (WS-NN.K)" | ...}

by = {r["mid"]: dict(r) for r in reg}
cluster_of = {i: c["name"] for c in syn["clusters"] for i in c["ids"]}
groups = syn.get("possible_overlaps") or []

# ---- severity re-grade (applied before merging so a merged root takes the max) ----
regrade_log = []
for g in regr.get("regrades", []):
    i = g.get("id")
    if i in by and g.get("new") in RANK and g["new"] != by[i]["sev"]:
        regrade_log.append({"id": i, "old": by[i]["sev"], "new": g["new"], "rationale": g.get("rationale", "")})
        by[i]["prev_sev"] = by[i]["sev"]
        by[i]["sev"] = g["new"]

# ---- merges (union-find over AGY decisions, validated against the group membership) ----
parent = {i: i for i in by}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


merge_log, merge_rejected = [], []
for k, dec in enumerate(over):
    gid = dec.get("group") or f"O-{k + 1:02d}"
    try:
        members = set(groups[int(gid.split("-")[1]) - 1]["ids"])
    except Exception:
        members = set()
    if dec.get("decision") not in ("MERGE", "PARTIAL_MERGE"):
        continue
    canon = dec.get("canonical")
    absorbed = [a for a in dec.get("absorbed") or [] if a != canon]
    bad = [x for x in [canon] + absorbed if x not in members or x not in by]
    if bad or not absorbed:
        merge_rejected.append({"group": gid, "reason": f"invalid ids {bad}" if bad else "nothing absorbed"})
        continue
    for a in absorbed:
        ra, rc = find(a), find(canon)
        if ra != rc:
            parent[ra] = rc
    merge_log.append({"group": gid, "canonical": canon, "absorbed": absorbed, "reason": dec.get("reason", "")})

components = collections.defaultdict(list)
for i in by:
    components[find(i)].append(i)


def pick_root(ids):
    return sorted(ids, key=lambda i: (-RANK[norm(by[i]["sev"])], int(i.split("-")[1])))[0]


entries = {}
absorbed_into = {}
for _, ids in components.items():
    root = pick_root(ids)
    e = dict(by[root])
    e["absorbed"] = sorted([i for i in ids if i != root], key=lambda i: int(i.split("-")[1]))
    e["sev"] = max((norm(by[i]["sev"]) for i in ids), key=lambda s: RANK[s])
    e["sources"] = sorted({s for i in ids for s in by[i]["ids"]})
    e["cluster"] = cluster_of.get(root, "Other")
    e["trackers"] = []
    e["origin"] = "review"
    entries[root] = e
    for i in e["absorbed"]:
        absorbed_into[i] = root

# ---- added entries: scope re-grade gaps, then verified tracker gaps ----
next_id = max(int(i.split("-")[1]) for i in by) + 1
added = []
for src, lst in (("scope re-grade", regr.get("new_entries", [])), ("tracker gap", gaps.get("entries", []))):
    for ne in lst:
        mid = f"MVP-{next_id:03d}"
        next_id += 1
        e = {"mid": mid, "sev": norm(ne.get("severity")), "area": ne.get("area", "misc"), "cat": ne.get("category", "missing-feature"),
             "title": ne.get("title"), "evidence": ne.get("evidence"), "scenario": ne.get("failure_scenario", ""),
             "ids": [], "sources": [ne["tracker_id"]] if ne.get("tracker_id") else [], "absorbed": [],
             "cluster": ne.get("cluster") or ("Enterprise basics: backups, MFA, audit trail, secrets" if src == "scope re-grade" else "Other"),
             "trackers": [ne["tracker_id"]] if ne.get("tracker_id") else [], "origin": src, "verif": f"Added by {src}; Opus-verified",
             "rationale": ne.get("rationale", "")}
        entries[mid] = e
        added.append(e)

# ---- tracker cross-reference ----
xref_rows = []
for x in xref:
    tid = f"{x.get('tracker')}:{x.get('id')}"
    mids = []
    for m in x.get("mvp_ids") or []:
        m = absorbed_into.get(m, m)
        if m in entries and m not in mids:
            mids.append(m)
    for m in mids:
        if tid not in entries[m]["trackers"]:
            entries[m]["trackers"].append(tid)
    xref_rows.append({**x, "mvp_ids": mids})
for g in gaps.get("entries", []):  # gap entries resolve their tracker rows
    for r in xref_rows:
        if f"{r.get('tracker')}:{r.get('id')}" == g.get("tracker_id"):
            mid = next(e["mid"] for e in added if g.get("tracker_id") in e["trackers"])
            r["mvp_ids"] = [mid]
            r["coverage"] = "full (added)"

# execution status: derived from the persistent tracker docs/mvp/TRACKER.md (task Status cells);
# status.json stays as a manual per-entry override applied last
TRACKER = os.path.join(os.path.dirname(DOC), "mvp", "TRACKER.md")
task_status = {}
if os.path.exists(TRACKER):
    for line in open(TRACKER, encoding="utf-8"):
        m = re.match(r"^\|\s*\[?(WS-\d{2}\.[A-Za-z0-9]+)\]?", line)
        if m:
            cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
            task_status[m.group(1)] = cells[-2].strip("`") if len(cells) >= 2 else "todo"

for e in entries.values():
    e["plan_tasks"] = plan_tasks.get(e["mid"], [])
    e.setdefault("status", "open")
    ts = [task_status.get(t, "todo") for t in e["plan_tasks"]]
    if ts and all(s == "done" for s in ts):
        e["status"] = f"fixed ({', '.join(e['plan_tasks'])})"
    elif any(s == "blocked" for s in ts):
        e["status"] = f"blocked ({', '.join(t for t in e['plan_tasks'] if task_status.get(t) == 'blocked')})"
    elif any(s not in ("todo", "deferred") for s in ts):
        e["status"] = f"in progress ({sum(1 for s in ts if s == 'done')}/{len(ts)} tasks done)"
    if e["mid"] in status_over:
        e["status"] = status_over[e["mid"]]

ordered = sorted(entries.values(), key=lambda e: (-RANK[e["sev"]], e["cluster"], int(e["mid"].split("-")[1])))
json.dump(ordered, open(os.path.join(P, "register_compiled.json"), "w", encoding="utf-8"), indent=1)
with open(os.path.join(P, "register_compact.txt"), "w", encoding="utf-8") as fh:
    fh.write("# id | severity | area | cluster | category | title | evidence\n")
    for e in ordered:
        fh.write(f"{e['mid']} | {e['sev']} | {e['area']} | {e['cluster']} | {e['cat']} | {clip(e['title'], 220)} | {clip(e.get('evidence'), 200)}\n")

# ---- render ----
L = []
w = L.append
cnt = collections.Counter(e["sev"] for e in ordered)
w("# Solace-AI MVP Issue Register")
w("")
w("This is the single list of open blockers, issues and bugs for the MVP release. Planning and execution track work against these IDs. Compiled 2026-10-18 from:")
w("")
w("- the verified review in `docs/MVP-BUG-REVIEW.md` (344 entries, each independently verified against the code);")
w("- a re-grade against the confirmed scope;")
w("- a merge of duplicate entries;")
w("- a cross-reference of every earlier tracker: `ENTERPRISE-READINESS-ASSESSMENT.md`, `FUNCTIONAL-COMPLETENESS.md`, the 2026-10-11 full-picture report, `BUG-BACKLOG.md`, `PHASE-C-SECURITY-FINDINGS.md`, `KNOWN-LIMITATIONS.md` and `POST-MVP-BACKLOG.md`.")
w("")
w("IDs are stable. A merged entry keeps its ID in the \"Absorbed\" list of the entry that now tracks it. Evidence cites the working tree as of 2026-10-18; no code has changed since the review.")
w("")
w("## 1. Scope and severity")
w("")
w(SCOPE)
w("")
w("Severity levels:")
w("")
w("- **blocker:** cannot ship, including a required control that is absent or does not work.")
w("- **high:** fix before release unless the owner explicitly accepts it.")
w("- **medium:** should fix for release or shortly after.")
w("- **low:** minor.")
w("- **post-mvp:** real, but outside the confirmed scope.")
w("")
w("## 2. Summary")
w("")
w("| blocker | high | medium | low | post-mvp | total open |")
w("|---|---|---|---|---|---|")
w("| " + " | ".join(str(cnt[s]) for s in SEVS[:5]) + f" | {len(ordered)} |")
w("")
w(f"Compared with the review's 344 entries: {len(merge_log)} merge decisions absorbed {sum(len(e['absorbed']) for e in ordered)} entries, the scope re-grade changed {len(regrade_log)} severities, and {len(added)} entries were added.")
w("")
w("**By root-cause cluster**")
w("")
w("| Cluster | blocker | high | medium | low | post-mvp | total |")
w("|---|---|---|---|---|---|---|")
by_c = collections.defaultdict(collections.Counter)
for e in ordered:
    by_c[e["cluster"]][e["sev"]] += 1
for c in sorted(by_c, key=lambda c: (-by_c[c]["blocker"], -by_c[c]["high"], c)):
    w(f"| {cell(c)} | " + " | ".join(str(by_c[c][s]) for s in SEVS[:5]) + f" | {sum(by_c[c].values())} |")
w("")
w("**By area**")
w("")
w("| Area | blocker | high | medium | low | post-mvp | total |")
w("|---|---|---|---|---|---|---|")
by_a = collections.defaultdict(collections.Counter)
for e in ordered:
    by_a[e["area"]][e["sev"]] += 1
for a in sorted(by_a, key=lambda a: (-by_a[a]["blocker"], -by_a[a]["high"], a)):
    w(f"| {a} | " + " | ".join(str(by_a[a][s]) for s in SEVS[:5]) + f" | {sum(by_a[a].values())} |")
w("")
w("## 3. Register")
w("")
w("Columns:")
w("")
w("- **Sources:** F-xxx are findings from the earlier Fable review; N-xxx are findings from the 2026-10 hunt.")
w("- **Trackers:** IDs in earlier trackers that describe the same issue.")
w("- **Plan task:** the task in the remediation plan that fixes it.")
w("")
titles = {"blocker": "3.1 Blockers", "high": "3.2 High", "medium": "3.3 Medium", "low": "3.4 Low", "post-mvp": "3.5 Post-MVP (outside the confirmed scope)"}
for s in SEVS[:5]:
    rows = [e for e in ordered if e["sev"] == s]
    w(f"### {titles[s]} ({len(rows)})")
    w("")
    if not rows:
        w("_none_")
        w("")
        continue
    w("| ID | Area | Cluster | Issue | Failure scenario | Evidence | Sources / absorbed | Trackers | Plan task | Status |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for e in rows:
        src = ", ".join(e.get("sources") or []) + (("; absorbed " + ", ".join(e["absorbed"])) if e["absorbed"] else "")
        w(f"| {e['mid']} | {e['area']} | {cell(e['cluster'])} | {cell(e['title'])} | {cell(clip(e.get('scenario'), 420))} | "
          f"{cell(clip(e.get('evidence'), 260))} | {cell(src)} | {cell(', '.join(e['trackers']))} | {cell(', '.join(e['plan_tasks'])) or '—'} | {e['status']} |")
    w("")
w("## 4. Changes made while compiling")
w("")
w(f"### 4.1 Scope re-grade ({len(regrade_log)} changes)")
w("")
if regrade_log:
    w("| ID | Old | New | Why |")
    w("|---|---|---|---|")
    for g in regrade_log:
        w(f"| {g['id']} | {g['old']} | {g['new']} | {cell(clip(g['rationale'], 400))} |")
else:
    w("_none_")
w("")
cov = regr.get("control_coverage") or []
if cov:
    w("**Coverage of the four enterprise basics**")
    w("")
    w("| Control | Covered by | Gaps (now added as entries) |")
    w("|---|---|---|")
    for c in cov:
        ids = [absorbed_into.get(i, i) for i in c.get("covered_by") or []]
        w(f"| {cell(c.get('control'))} | {', '.join(dict.fromkeys(ids)) or '—'} | {cell(clip(c.get('gaps'), 400)) or '—'} |")
    w("")
w(f"### 4.2 Merged duplicates ({len(merge_log)} merges)")
w("")
if merge_log:
    w("| Group | Kept | Absorbed | Why |")
    w("|---|---|---|---|")
    for m in merge_log:
        w(f"| {m['group']} | {absorbed_into.get(m['canonical'], m['canonical'])} | {', '.join(m['absorbed'])} | {cell(clip(m['reason'], 300))} |")
else:
    w("_none_")
w("")
if merge_rejected:
    w("Merge decisions rejected by validation: " + "; ".join(f"{r['group']} ({r['reason']})" for r in merge_rejected))
    w("")
w(f"### 4.3 Added entries ({len(added)})")
w("")
if added:
    w("| ID | Severity | Origin | Issue | Why |")
    w("|---|---|---|---|---|")
    for e in added:
        w(f"| {e['mid']} | {e['sev']} | {e['origin']} | {cell(e['title'])} | {cell(clip(e.get('rationale'), 300))} |")
else:
    w("_none_")
w("")
w(f"## 5. Earlier trackers cross-reference ({len(xref_rows)} items)")
w("")
w("Each row maps an item from an earlier tracker to the register entries that now track it. \"none\" with status \"fixed\" means the earlier tracker reports it closed.")
w("")
if xref_rows:
    w("| Tracker | ID | Title | Status there | Register IDs | Coverage | Note |")
    w("|---|---|---|---|---|---|---|")
    for x in xref_rows:
        w(f"| {x.get('tracker')} | {x.get('id')} | {cell(clip(x.get('title'), 120))} | {x.get('status_in_tracker')} | {', '.join(x['mvp_ids']) or '—'} | {x.get('coverage')} | {cell(clip(x.get('note'), 240))} |")
    w("")
rej = gaps.get("rejected") or []
if rej:
    w("**Tracker items checked and not added** (fixed, not reproducible, or out of scope):")
    w("")
    for r in rej:
        w(f"- {r.get('tracker_id')}: {cell(clip(r.get('reason'), 300))}")
    w("")
open(DOC, "w", encoding="utf-8").write("\n".join(L) + "\n")
print(f"entries {len(ordered)}; by severity {dict(cnt)}; merges {len(merge_log)} (rejected {len(merge_rejected)}); regrades {len(regrade_log)}; added {len(added)}; xref {len(xref_rows)}")
print("wrote", DOC)
