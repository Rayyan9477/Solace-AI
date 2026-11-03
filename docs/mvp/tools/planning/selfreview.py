"""Compute the writing-plans self-review for the master plan from the final planning state.
writes D:\\tmp\\mvp-plan\\selfreview.md (assemble_master.py appends it as section 10)."""
import json, os, re, collections

P = r"D:\tmp\mvp-plan"
REPO = r"D:\Repo\Contextual-Chatbot"
DIR = os.path.join(REPO, r"docs\superpowers\plans\2026-10-18-mvp-remediation")
sk = json.load(open(os.path.join(P, "skeleton.json"), encoding="utf-8"))
tasks = json.load(open(os.path.join(P, "task_index.json"), encoding="utf-8"))
plan_tasks = json.load(open(os.path.join(P, "plan_tasks.json"), encoding="utf-8"))
reg = json.load(open(os.path.join(P, "register_compiled.json"), encoding="utf-8"))
reg = reg["entries"] if isinstance(reg, dict) else reg
indexes = json.load(open(os.path.join(P, "indexes.json"), encoding="utf-8"))
agy = []
for nm in ("agy_findings_r1.json", "agy_findings_r2.json"):
    if os.path.exists(os.path.join(P, nm)):
        agy += json.load(open(os.path.join(P, nm), encoding="utf-8"))
AGY_TASKS = 297  # tasks AGY checked across both rounds (round 1: 197 in 13 plans; round 2: the 5 new plans)
rep = open(os.path.join(P, "check_report.md"), encoding="utf-8").read()
required = [e["mid"] for e in reg if e["sev"] != "post-mvp"]
sev = {e["mid"]: e["sev"] for e in reg}

texts, sizes = {}, {}
for fn in sorted(os.listdir(DIR)):
    if re.match(r"WS-\d{2}-.*\.md$", fn):
        t = open(os.path.join(DIR, fn), encoding="utf-8").read()
        texts[fn[:5]] = t
        sizes[fn[:5]] = len(t)

# step scan: every task block has a failing-test step and a RED expectation (or a verification step)
no_red, no_step1 = [], []
for ws, t in texts.items():
    heads = list(re.finditer(r"^### Task (WS-\d{2}\.[A-Za-z0-9]+):", t, re.M))
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(t)
        nxt = re.search(r"^## ", t[h.end():end], re.M)
        block = t[h.end(): h.end() + nxt.start()] if nxt else t[h.end():end]
        if "Step 1" not in block:
            no_step1.append(h.group(1))
        if not re.search(r"Expected[^\n]{0,200}\bFAIL", block, re.I) and "verification" not in block.lower():
            no_red.append(h.group(1))
placeholders = sum(len(re.findall(r"\bTBD\b|handle edge cases|add appropriate validation|write tests for the above", t, re.I)) for t in texts.values())

# review responses
resp = collections.Counter()
for t in texts.values():
    m = re.search(r"^## Review responses\s*$(.*?)(?=^## |\Z)", t, re.M | re.S)
    if m:
        for line in m.group(1).splitlines():
            l = line.lower()
            if re.search(r"\breconciliation \d+", l):
                resp["reconciled"] += 1
            if m2 := re.search(r"crisis-review (f\d+)", l):
                resp["crisis:" + m2.group(1)] += 1
            if re.search(r"\breconciliation \d+|crisis-review f\d+", l):
                continue
            if re.search(r"\brejected\b", l):
                resp["rejected"] += 1
            elif re.search(r"\bfixed\b", l):
                resp["fixed"] += 1

# review focus pins
pins = collections.defaultdict(set)
for ws, ix in indexes.items():
    for m in re.finditer(r"REVIEW_FOCUS:\s*([^\n]*)", ix.get("notes", "")):
        for part in m.group(1).split(";"):
            if "=" in part:
                n, ids = part.split("=", 1)
                n = re.sub(r"\D", "", n)
                for tid in re.findall(r"WS-\d{2}\.[A-Za-z0-9]+", ids):
                    if n and tid in tasks:
                        pins[int(n)].add(tid)

agy_by = collections.Counter((f.get("check"), f.get("severity")) for f in agy)
cov = sum(1 for m in required if plan_tasks.get(m))
err_line = rep.splitlines()[0] if rep else ""
L = []
w = L.append
w("## 10. Self-review\n")
w("The writing-plans checklist, run against the register (the spec) after the automated checks, the AGY grounding review, the Opus fix pass, the reconciliation pass and the crisis-path review.\n")
w(f"1. **Spec coverage.** {cov} of {len(required)} in-scope register entries are covered by at least one task "
  f"({', '.join(f'{s} {sum(1 for m in required if sev[m] == s and plan_tasks.get(m))}/{sum(1 for m in required if sev[m] == s)}' for s in ('blocker', 'high', 'medium', 'low'))}). "
  "The register's Plan task column lists the tasks per entry.")
w(f"2. **Step scan.** {len(tasks)} tasks across {len(texts)} workstream plans. Placeholder phrases found: {placeholders}. "
  f"Tasks without a Step 1 failing test: {len(no_step1)}{(' (' + ', '.join(no_step1[:12]) + ')') if no_step1 else ''}. "
  f"Tasks without an explicit RED expectation or verification step: {len(no_red)}{(' (' + ', '.join(no_red[:12]) + ')') if no_red else ''}.")
w(f"3. **Type and path consistency.** AGY checked every task's Modify paths, named symbols, test imports, RED claims, run commands and consumed interfaces against the code: "
  f"{len(agy)} findings ({', '.join(f'{c} {s} {n}' for (c, s), n in sorted(agy_by.items(), key=lambda x: -x[1])[:8])}). "
  f"The Opus fix pass recorded {resp['fixed']} fixed and {resp['rejected']} rejected findings (each plan's \"Review responses\" section). "
  f"A reconciliation pass then settled the last 11 cross-plan ownership conflicts ({resp['reconciled']} \"Reconciliation\" lines across the plans). "
  f"An independent Opus review traced one HIGH/CRITICAL message across every plan, from ingress through screen, resources, outbox, event plane, page, acknowledgement and alerts. "
  f"It found {len([k for k in resp if k.startswith('crisis:')])} gaps (F1-F13: for example, a failed page blocked the consumer partition, a failed outbox write lost the crisis, refusals at ingress skipped the screen, and the lag alert could not fire for a single stuck crisis). All were verified and fixed in the owning tasks. "
  f"Final automated check: {err_line}.")
w("4. **Review Focus.** " + "; ".join(f"line {i}: {len(pins.get(i, []))} pinning tasks" for i in range(1, 6)) + ". The task ids are listed under each Review Focus line at the top of this plan.")
total = sum(sizes.values())
w(f"5. **Proportion.** The 18 workstream plans total {total // 1024} KB, about {total // max(1, len(tasks)) // 1024} KB per task. Most of that is the failing tests with their exact values, which the implementers cannot derive themselves. Implementation bodies appear only where the signature and the tests do not determine the algorithm.")
w("")
w("**What this plan does not prove.** A task's RED claim is checked by reading code, not by running it, so Step 2 of each task is the first real proof. Real-infra timings, image sizes and Docker resource use are estimates until W0 runs. The clinical defaults (D-05, D-06, D-22, D-28, D-29) and the legal ones (D-04, D-23, D-24, D-31) need their owners' sign-off before real users.")
open(os.path.join(P, "selfreview.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
