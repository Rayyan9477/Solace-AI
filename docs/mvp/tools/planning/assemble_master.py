"""Assemble the master plan and the interfaces/ownership appendix from the planning state.

inputs (D:\\tmp\\mvp-plan): skeleton.json, task_index.json, schedule.json, plan_tasks.json, indexes.json,
                           register_compiled.json, master_protocol.md, selfreview.md (optional)
outputs: docs/superpowers/plans/2026-10-18-mvp-remediation.md
         docs/superpowers/plans/2026-10-18-mvp-remediation/00-interfaces-and-ownership.md
env MASTER_OUT / APPX_OUT override the output paths (dry runs).
"""
import json, os, re, collections

P = r"D:\tmp\mvp-plan"
REPO = r"D:\Repo\Contextual-Chatbot"
MASTER = os.environ.get("MASTER_OUT", os.path.join(REPO, r"docs\superpowers\plans\2026-10-18-mvp-remediation.md"))
APPX = os.environ.get("APPX_OUT", os.path.join(REPO, r"docs\superpowers\plans\2026-10-18-mvp-remediation\00-interfaces-and-ownership.md"))
SUB = "2026-10-18-mvp-remediation"


def load(name, default):
    p = os.path.join(P, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else default


sk = load("skeleton.json", None)
tasks = load("task_index.json", {})
sched = load("schedule.json", {})
plan_tasks = load("plan_tasks.json", {})
indexes = load("indexes.json", {})
reg = load("register_compiled.json", [])
reg = reg["entries"] if isinstance(reg, dict) else reg
proto = open(os.path.join(P, "master_protocol.md"), encoding="utf-8").read()
selfreview = open(os.path.join(P, "selfreview.md"), encoding="utf-8").read() if os.path.exists(os.path.join(P, "selfreview.md")) else ""

sev = {e["mid"]: e["sev"] for e in reg}
required = [e["mid"] for e in reg if e["sev"] != "post-mvp"]
ws_list = sk["workstreams"]
ws_by_id = {w["id"]: w for w in ws_list}
SEVS = ["blocker", "high", "medium", "low"]


def cell(s):
    return str(s).replace("|", "\\|").replace("\n", " ").strip()


def plan_file(ws):
    ix = indexes.get(ws)
    if ix:
        return ix["file"].split("/")[-1]
    w = ws_by_id[ws]
    return f"{w['id']}-{w['slug']}.md"


def tlink(tid):
    t = tasks.get(tid)
    return f"[{tid}]({SUB}/{plan_file(t['ws'])})" if t else tid


def sev_counts(ids):
    c = collections.Counter(sev.get(i, "?") for i in ids)
    return "/".join(str(c[s]) for s in SEVS)


# ---------- review-focus pins from planner notes ----------
pins = collections.defaultdict(set)
for ws, ix in indexes.items():
    for m in re.finditer(r"REVIEW_FOCUS:\s*([^\n]*)", ix.get("notes", "")):
        for part in m.group(1).split(";"):
            if "=" in part:
                n, ids = part.split("=", 1)
                n = re.sub(r"\D", "", n)
                for tid in re.findall(r"WS-\d+\.\w+", ids):
                    if n and tid in tasks:
                        pins[int(n)].add(tid)

out = []
w = out.append

# ---------- header ----------
w("# Solace-AI MVP Remediation Implementation Plan\n")
w("> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development, as adapted in section 5 of this plan (no git commits, parallel dispatch of file-disjoint tasks, AGY as a third reviewer). Steps use checkbox (`- [ ]`) syntax for tracking.\n")
w(f"**Goal:** Make the Solace-AI backend releasable as the confirmed \"Pilot + enterprise basics\" MVP. Close all {len(required)} in-scope entries of the issue register, each proven by a test that fails before the fix, then pass the release gate twice on the strict production profile.\n")
w("**Architecture:** Six barrier waves (W0 to W5) over 18 workstreams with disjoint file ownership, executed in parallel in one working tree with commit-free checkpoints. Contention is removed by seams rather than by scheduling: one runtime bootstrap that every service `main.py` adopts once, pre-allocated Alembic slot revisions, single owners for compose, the environment contract and the seven event bridges, a one-time user-service split, one clinician-reviewed crisis lexicon for all five detectors, and short-lived on-behalf-of service tokens so no user credential enters LangGraph state. Waves follow risk: build, migrate and boot; the crisis path; the chat turn and durable state; privacy controls; the enterprise basics; then residual defects and the release gate.\n")
w("**Tech Stack:** Python 3.12, FastAPI, LangGraph 1.0 with a Postgres checkpointer, PostgreSQL 16 with row-level security, Kafka with a transactional outbox, Redis, Weaviate, Caddy, Docker Compose, Alembic, a uv hash-pinned lock, pytest with testcontainers; Anthropic models on AWS Bedrock for LLM features (D-04), Amazon SES for clinician alerts (D-12), SOPS + age for secrets (D-07), AWS us-east-1 (EC2 via SSM, S3 with KMS) for the release gate and the pilot (D-25).\n")
w(f"**Spec:** [docs/MVP-ISSUE-REGISTER.md](../../MVP-ISSUE-REGISTER.md) is the binding issue list and records the confirmed scope; [docs/MVP-BUG-REVIEW.md](../../MVP-BUG-REVIEW.md) is its verification record. Every task is in one of the 18 workstream plans in [{SUB}/]({SUB}/). Interface contracts and file ownership are in [00-interfaces-and-ownership.md]({SUB}/00-interfaces-and-ownership.md).\n")
w("**Tracking:** [docs/mvp/README.md](../../mvp/README.md) is the hub. It links [DECISIONS.md](../../mvp/DECISIONS.md) (owner decisions and sign-offs), [TRACKER.md](../../mvp/TRACKER.md) (per-task status and the barrier log), [OPEN-ITEMS.md](../../mvp/OPEN-ITEMS.md) (owner actions, sign-offs, risks) and [LOG.md](../../mvp/LOG.md) (dated history). Those files are the persistent record; this plan is the method.\n")

w("## Global Constraints\n")
for g in sk["global_constraints"]:
    g = g.replace("in D:\\tmp\\mvp-plan\\register_compiled.json", "in D:\\tmp\\mvp-plan\\status.json (compile_register.py applies it)")
    w(f"- {g}")
extra = [
    "Execution starts only on the owner's \"go\". Until then nothing in the repo changes except these plan documents and the register.",
    "Secrets use SOPS + age (D-07); no AWS Secrets Manager or Parameter Store holds secrets. If any AWS secrets work ever arises, load the aws-secrets-manager skill first and use `{{resolve:secretsmanager:...}}` references through asm-exec; no agent calls `get-secret-value` or `batch-get-secret-value`, and no secret value or age private key enters a prompt, report, log or the repo.",
    "Decisions are recorded in docs/mvp/DECISIONS.md, the live record; section 3 is the snapshot of 2026-10-19. A changed decision is a ledger Ruling plus an entry in DECISIONS.md and docs/mvp/LOG.md before any task acts on it.",
    "AWS resource names and descriptions use hyphens, never em dashes. AWS work prefers the AWS MCP server and infrastructure as code (CloudFormation in infrastructure/aws-staging).",
]
for g in extra:
    w(f"- {g}")
w("")

w("## Review Focus\n")
w("These five failure modes are what the register implies but no single defect fix exercises. Each is pinned by the tests of the tasks named after it.\n")
for i, rf in enumerate(sk["review_focus"], 1):
    p = ", ".join(tlink(t) for t in sorted(pins.get(i, []), key=lambda x: (x.split(".")[0], int(re.sub(r"\D", "", x.split(".")[1]) or 0))))
    w(f"{i}. {rf}" + (f" **Pinned by:** {p}." if p else " **Pinned by:** (none yet)."))
w("\n---\n")

# ---------- 1. how to use ----------
w("## 1. How to use this plan\n")
w("- **Owner:** sections 2 to 4 are the scope, your decisions (answered on 2026-10-19) and the waves. Progress, open owner actions and sign-offs are tracked in [docs/mvp/](../../mvp/README.md).")
w("- **Controller:** section 5 is the execution protocol. Run waves strictly in order. Inside a wave, run phase A tasks (platform) before phase B tasks (services), and dispatch each task when its dependencies are done and its files and suites are free (section 5.3).")
w(f"- **Implementers and reviewers:** each task `WS-NN.K` is the block `### Task WS-NN.K:` in its workstream plan under [{SUB}/]({SUB}/). `D:\\tmp\\mvp-exec\\tools\\extract-brief.ps1 -Task WS-NN.K` copies one block to a brief. An implementer reads only its brief and the files it names; the interface names it must use are in the brief's Interfaces block and in the appendix.\n")

# ---------- 2. scope and coverage ----------
w("## 2. Scope and coverage\n")
w("**Confirmed scope (owner, 2026-10-18): \"Pilot + enterprise basics\".** A single-region pilot with real users on one docker-compose host: email/password and Google sign-in, chat through the orchestrator REST and WebSocket API, every message safety-screened, a detected crisis shows resources and alerts an on-call clinician, durable PHI that is encrypted and isolated per user, working account deletion, health checks and basic monitoring and alerting. Plus four enterprise basics: backups with a tested restore, MFA, a working audit trail, and per-environment secrets management. Out of scope: HA/DR across zones, horizontal scaling, load and chaos testing, SOC2 evidence, the Kong gateway package, ClickHouse analytics, config_service as a product feature, and an enterprise RBAC/ABAC engine.\n")
cov = sum(1 for m in required if plan_tasks.get(m))
w(f"**Coverage:** {cov} of {len(required)} in-scope register entries are covered by at least one task. The register's *Plan task* column names them. The 11 post-MVP entries stay in the register's backlog section.\n")
w("| Severity | In-scope entries | Covered |")
w("|---|---|---|")
for s in SEVS:
    ids = [m for m in required if sev[m] == s]
    w(f"| {s} | {len(ids)} | {sum(1 for m in ids if plan_tasks.get(m))} |")
w("")
if tasks:
    first_wave = {}
    for m in required:
        tw = [tasks[t]["wave"] for t in plan_tasks.get(m, []) if t in tasks]
        if tw:
            first_wave[m] = min(tw)
    w("| Wave | Tasks (phase A / B) | Critical path A / B | Dispatch rounds A / B (6 implementers) | Real-infra tasks | Opus / Sonnet implementers | Entries first fixed here (B/H/M/L) |")
    w("|---|---|---|---|---|---|---|")
    tot_r = 0
    for wv in sorted(sched, key=int):
        ts = [t for t in tasks.values() if str(t["wave"]) == wv]
        mc = collections.Counter(t["model"] for t in ts)
        ids = [m for m, x in first_wave.items() if str(x) == wv]
        sa, sb = sched[wv].get("A", {}), sched[wv].get("B", {})
        tot_r += sa.get("rounds", 0) + sb.get("rounds", 0)
        w(f"| W{wv} | {len(ts)} ({sa.get('tasks', 0)} / {sb.get('tasks', 0)}) | {sa.get('critical_path', 0)} / {sb.get('critical_path', 0)} | "
          f"{sa.get('rounds', 0)} / {sb.get('rounds', 0)} | {sum(1 for t in ts if t['real_infra'])} | {mc['opus']} / {mc['sonnet']} | {sev_counts(ids)} |")
    w(f"\nA dispatch round is one batch of concurrent tasks whose dependencies are done and whose files and suites do not overlap; the {len(tasks)} tasks need about {tot_r} rounds in total, before review and fix rounds.\n")

# ---------- 3. decisions ----------
w("## 3. Owner decisions\n")
w("The owner answered all 31 decisions on 2026-10-19. Every task follows the *Decided* line below and names the decision in its *Decision gate* line. Six answers changed the recommended default (D-04, D-07, D-17, D-20, D-25, D-29), and the workstream plans were updated to match (each plan's Review responses lists the changes as \"decision 2026-10-19\"). Clinical and legal items are built on their defaults now. Their sign-offs, and the actions that need an outside party (BAA, IAM policy, on-call names), do not block engineering, but they block go-live with real users (D-31). The live record is [docs/mvp/DECISIONS.md](../../mvp/DECISIONS.md); open owner actions are in [docs/mvp/OPEN-ITEMS.md](../../mvp/OPEN-ITEMS.md).\n")
st_count = collections.Counter(d.get("status", "open") for d in sk["decisions"])
w("| Status | Decisions |")
w("|---|---|")
for stv in ("Decided (default)", "Decided (changed from default)", "Default accepted, sign-off pending"):
    w(f"| {stv} | {', '.join(d['id'] for d in sk['decisions'] if d.get('status') == stv)} |")
w("")
w("Saying \"go\" starts W0 with these answers. W0 entry creates the program venv (D-20, section 5.0); nothing is installed into anaconda and nothing is committed (D-17).\n")
by_wave = collections.defaultdict(list)
for d in sk["decisions"]:
    by_wave[d.get("needed_by_wave")].append(d)
for wv in sorted(by_wave, key=lambda x: (x is None, x)):
    w(f"### Needed by W{wv}\n")
    for d in by_wave[wv]:
        w(f"- **{d['id']} ({d.get('owner')}): {d['question']}**")
        if d.get("answer"):
            w(f"  - *Decided ({d.get('answered_on', '')}, {d.get('status', '')}):* {d['answer']}")
        lbl = "Original recommendation (replaced)" if "changed" in d.get("status", "") else "Default"
        w(f"  - *{lbl}:* {d['recommended_default']}")
        if d.get("rationale"):
            w(f"  - *Why:* {d['rationale']}")
        if d.get("affects"):
            aff = d["affects"] if isinstance(d["affects"], str) else ", ".join(d["affects"])
            w(f"  - *Affects:* {aff}")
    w("")

# ---------- 4. waves ----------
w("## 4. Waves, phases and tasks\n")
w("A wave ends at a barrier (section 5.5). Inside a wave, phase A tasks touch platform paths (`src/**`, `services/shared/**`, `migrations/**`, `infrastructure/**`, `deploy/**`, `docker/**`, the seven `event_bridge.py` files, compose, CI, build and requirements files) and run first. Phase B tasks touch service paths and start when phase A is done. The tables list each phase's tasks by dependency level: a task needs only tasks from earlier levels of its phase (and anything from earlier phases and waves). Tasks in one level run concurrently when their files and suites do not overlap, as section 5.3 describes.\n")
for wv in sk["waves"]:
    n = str(wv["wave"])
    w(f"### W{n}: {wv['name']}\n")
    w(f"**Goal.** {wv['goal']}\n")
    w("**Entry criteria.**")
    for c in wv["entry_criteria"]:
        w(f"- {c}")
    w("\n**Exit criteria.**")
    for c in wv["exit_criteria"]:
        w(f"- {c}")
    w(f"\n**Bug-round focus.** {wv['bug_round_focus']}\n")
    for ph, title in (("A", "Phase A (platform, runs first)"), ("B", "Phase B (services, after phase A)")):
        s = sched.get(n, {}).get(ph)
        if not s:
            continue
        w(f"**{title}:** {s['tasks']} tasks, critical path {s['critical_path']}, about {s['rounds']} dispatch rounds.\n")
        w("| Level | Tasks (real-infra tasks marked RI) |")
        w("|---|---|")
        for i, lv in enumerate(s["levels"], 1):
            names = "<br>".join(f"{tlink(t)} {cell(tasks[t]['name'])}{' (RI)' if tasks[t]['real_infra'] else ''}" for t in lv)
            w(f"| {i} | {names} |")
        w("")

# ---------- 5. protocol ----------
p = proto.replace("## 4. Execution protocol", "## 5. Execution protocol")
p = re.sub(r"^### 4\.(\d)", r"### 5.\1", p, flags=re.M)
p = p.replace("Section 3 lists the lanes of each wave.", "Section 4 lists the lanes of each wave.")
p = p.replace("(listed in the wave's exit criteria in section 3)", "(listed in the wave's exit criteria in section 4)")
p = p.replace("The release gate in section 3 (the last wave's exit criteria).", "The release gate in section 4 (W5's exit criteria).")
p = p.replace("(see 4.4)", "(see 5.4)").replace("(section 4.6)", "(section 5.6)")
p = re.sub(r"\(section 4\.(\d)\)", r"(section 5.\1)", p).replace("master plan, section 5.", "section 5.")
w(p.rstrip() + "\n")

# ---------- 6. workstreams ----------
w("## 6. Workstreams\n")
w("| WS | Workstream | Plan | Waves | Tasks | Entries (B/H/M/L) | Risk class |")
w("|---|---|---|---|---|---|---|")
for x in ws_list:
    nt = sum(1 for t in tasks.values() if t["ws"] == x["id"])
    w(f"| {x['id']} | {cell(x['name'])} | [{plan_file(x['id'])}]({SUB}/{plan_file(x['id'])}) | {', '.join('W' + str(v) for v in x['waves'])} | {nt} | {sev_counts(x['issue_ids'])} | {x['risk']} |")
w("")
for x in ws_list:
    w(f"- **{x['id']} goal:** {x['goal']}")
w("")

# ---------- 7. test strategy ----------
w("## 7. Test strategy\n")
w(sk["test_strategy"].strip() + "\n")

# ---------- 8. risks ----------
w("## 8. Risks\n")
for r in sk.get("risks", []):
    w(f"- {r}")
w("")

# ---------- 9. interfaces ----------
w("## 9. Interfaces and file ownership\n")
w(f"The full contracts, the shared-file protocols and every workstream's owned paths are in [00-interfaces-and-ownership.md]({SUB}/00-interfaces-and-ownership.md).\n")
w("| IF | Interface | Owner | Consumers |")
w("|---|---|---|---|")
for i in sk["interfaces"]:
    w(f"| {i['id']} | {cell(i['name'])} | {i['owner_ws']} | {', '.join(i.get('consumers', []))} |")
w("")

# ---------- 10. self-review ----------
if selfreview:
    w(selfreview.rstrip() + "\n")

os.makedirs(os.path.dirname(MASTER), exist_ok=True)
open(MASTER, "w", encoding="utf-8").write("\n".join(out))

# ---------- appendix ----------
a = []
A = a.append
A("# Interfaces and file ownership\n")
A("> Appendix to the [MVP remediation plan](../2026-10-18-mvp-remediation.md). Implementers use these names, signatures, env vars, topics and headers exactly. A change to a file another workstream owns goes through that owner (the plan's cross-workstream requests).\n")
A("## Interfaces\n")
for i in sk["interfaces"]:
    A(f"### {i['id']}: {i['name']}\n")
    A(f"**Owner:** {i['owner_ws']}. **Consumers:** {', '.join(i.get('consumers', []))}.\n")
    A(i["contract"].strip() + "\n")
A("## Shared files\n")
A("| File | Owner | Needed by | Protocol |")
A("|---|---|---|---|")
for s in sk.get("shared_files", []):
    A(f"| `{s['path']}` | {s['owner_ws']} | {', '.join(s.get('needed_by', []))} | {cell(s.get('protocol', ''))} |")
A("\n## Owned paths\n")
for x in ws_list:
    A(f"### {x['id']}: {x['name']}\n")
    A(", ".join(f"`{p}`" for p in x["owned_paths"]) + "\n")
    if x.get("produces") or x.get("consumes"):
        A(f"Produces: {', '.join(x.get('produces', [])) or 'none'}. Consumes: {', '.join(x.get('consumes', [])) or 'none'}.\n")
os.makedirs(os.path.dirname(APPX), exist_ok=True)
open(APPX, "w", encoding="utf-8").write("\n".join(a))
print("master:", MASTER, os.path.getsize(MASTER), "bytes;", "appendix:", APPX, os.path.getsize(APPX), "bytes")
print("review-focus pins:", {k: sorted(v) for k, v in pins.items()})
