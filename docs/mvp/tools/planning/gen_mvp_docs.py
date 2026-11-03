"""One-time generator for docs/mvp/DECISIONS.md and docs/mvp/OPEN-ITEMS.md.

After the first run these files are hand-maintained: the generator refuses to overwrite an existing
file unless --force is given. Inputs: D:\\tmp\\mvp-plan\\skeleton.json (decisions with the owner's
answers, risks) and the workstream plans (for the per-decision plan impact).
usage: gen_mvp_docs.py [--force]
"""
import json, os, re, sys, collections

P = r"D:\tmp\mvp-plan"
REPO = r"D:\Repo\Contextual-Chatbot"
MVP = os.path.join(REPO, "docs", "mvp")
PLANS = os.path.join(REPO, "docs", "superpowers", "plans", "2026-10-18-mvp-remediation")
PLAN_REL = "../superpowers/plans/2026-10-18-mvp-remediation"
force = "--force" in sys.argv
sk = json.load(open(os.path.join(P, "skeleton.json"), encoding="utf-8"))
WSF = {fn[:5]: fn for fn in os.listdir(PLANS) if fn.startswith("WS-")}
TID = r"WS-\d{2}\.[A-Za-z0-9]+"


def cell(s):
    return " ".join(str(s).split()).replace("|", "\\|")


# plan impact per decision: Review-responses lines tagged "decision 2026-10-19 D-NN" (and gates)
impact = collections.defaultdict(set)
gates = collections.defaultdict(set)
for ws, fn in WSF.items():
    t = open(os.path.join(PLANS, fn), encoding="utf-8").read()
    for h in re.finditer(r"^### Task (" + TID + r"):", t, re.M):
        s = h.end(); m = re.search(r"^(### Task |## )", t[s:], re.M); b = t[s:s + (m.start() if m else len(t))]
        g = re.search(r"^\*\*Decision gate:\*\*(.*)$", b, re.M)
        if g:
            for d in set(re.findall(r"D-\d{2}", g.group(1))):
                gates[d].add(h.group(1))
    rr = re.search(r"^## Review responses\s*$(.*?)(?=^## |\Z)", t, re.M | re.S)
    if rr:
        for line in rr.group(1).splitlines():
            m = re.search(r"decision 2026-10-19 (D-\d{2})", line)
            if m:
                ids = set(re.findall(TID, line)) or {ws}
                impact[m.group(1)] |= ids


def tlink(tid):
    ws = tid[:5]
    return f"[{tid}]({PLAN_REL}/{WSF[ws]})" if ws in WSF and "." in tid else tid


def sortkey(x):
    a, _, b = x.partition(".")
    return (a, int(re.sub(r"\D", "", b) or 0), b)


OWNER_ACTIONS = {
    "D-04": ["X-02", "X-08"], "D-07": ["X-07"], "D-12": ["X-02", "X-03"], "D-13": ["X-05"], "D-17": ["X-10", "X-11", "X-12"],
    "D-25": ["X-01"], "D-29": ["X-04", "X-09"], "D-05": ["X-04"], "D-06": ["X-04"], "D-21": ["X-04"], "D-22": ["X-04"],
    "D-28": ["X-04"], "D-30": ["X-04"], "D-23": ["X-06"], "D-24": ["X-06"], "D-31": ["X-06"],
}

# ---------------- DECISIONS.md ----------------
L = []
w = L.append
w("# MVP owner decisions\n")
w("The live record of every owner decision for the MVP remediation program. The master plan's section 3 is a snapshot of these answers as of 2026-10-19; this file wins if they differ.\n")
w("**How to update.** Change a decision only with the owner's say-so:")
w("1. Edit its entry: the *Decision* line, its *Status*, and a new row in its *History* table.")
w("2. Add a dated line to [LOG.md](LOG.md).")
w("3. Have the controller record a ledger Ruling and update every task named under *Plan impact* before any of them runs.")
w("")
w("Sign-off fields are filled in when the named approver confirms, with the date and where the evidence lives.\n")
w("**Status values:** `Decided (default)` means the plan's recommendation was accepted. `Decided (changed from default)` means the owner chose differently and the plans were updated. `Default accepted, sign-off pending` means engineering builds the default now, and a clinical or legal approver must confirm it before go-live (D-31).\n")
w("## Summary\n")
w("| ID | Topic | Owner role | Needed by | Status | Decision (short) |")
w("|---|---|---|---|---|---|")
for d in sk["decisions"]:
    short = d.get("answer", "")
    short = short if len(short) < 170 else short[:167].rsplit(" ", 1)[0] + "..."
    w(f"| [{d['id']}](#{d['id'].lower()}) | {cell(d['question'])} | {d.get('owner')} | W{d.get('needed_by_wave')} | {d.get('status')} | {cell(short)} |")
w("")
c = collections.Counter(d.get("status") for d in sk["decisions"])
w("Counts: " + "; ".join(f"{k}: {v}" for k, v in c.items()) + ".\n")
w("## Decisions\n")
for d in sk["decisions"]:
    w(f"### {d['id']}\n")
    w(f"**Question.** {d['question']}\n")
    w(f"- **Status:** {d.get('status')} (answered {d.get('answered_on', '')} by the owner)")
    w(f"- **Owner role:** {d.get('owner')}; needed by W{d.get('needed_by_wave')}")
    w(f"- **Decision:** {d.get('answer', '')}")
    lbl = "Original recommendation (replaced)" if "changed" in (d.get("status") or "") else "Default (as built)"
    w(f"- **{lbl}:** {d['recommended_default']}")
    if d.get("rationale"):
        w(f"- **Why:** {d['rationale']}")
    aff = d.get("affects") or []
    aff = aff if isinstance(aff, str) else ", ".join(aff)
    w(f"- **Affects workstreams:** {aff or 'none listed'}")
    gi = sorted(gates.get(d["id"], []), key=sortkey)
    w(f"- **Tasks gated on it:** {', '.join(tlink(t) for t in gi) if gi else 'none'}")
    if d["id"] in impact:
        ii = sorted(impact[d["id"]], key=sortkey)
        w(f"- **Plan impact (2026-10-19 change):** {', '.join(tlink(t) for t in ii)}. Each plan's *Review responses* lists the exact change under \"decision 2026-10-19 {d['id']}\".")
    if d["id"] in OWNER_ACTIONS:
        w(f"- **Owner actions:** {', '.join(f'[{x}](OPEN-ITEMS.md#owner-actions)' for x in OWNER_ACTIONS[d['id']])}")
    if "sign-off" in (d.get("status") or ""):
        w("- **Sign-off:** approver: _(name)_; date: _(date)_; evidence: _(link)_")
    w("\n| Date | Change | By |")
    w("|---|---|---|")
    w(f"| 2026-10-18 | Recommended default written in the plan | planning |")
    w(f"| {d.get('answered_on', '')} | {d.get('status')}: {cell(d.get('answer', ''))[:200]} | owner |")
    w("")
DEC = "\n".join(L) + "\n"

# ---------------- OPEN-ITEMS.md ----------------
X = [
    ("X-01", "Attach `infrastructure/aws-staging/solace-staging-deployer-policy.json` to the existing IAM user `MedCodex-Local` (option 2 in AWS-STAGING-IAM-SETUP.pdf).", "D-25, D-09, D-12, D-04", "W4 (off-host backup target), W5 (release gate), go-live", "open", "IAM simulation on 2026-10-19: only `bedrock:InvokeModel` and `bedrock:ListFoundationModels` allowed; CloudFormation, EC2, SSM, IAM role creation, S3, KMS and SES denied."),
    ("X-02", "Accept the AWS BAA in AWS Artifact and record the date here.", "D-04, D-12", "go-live (any real-user PHI to Bedrock or in SES alerts)", "open", "Not confirmed yet. Until it is recorded, LLM features stay off for real users."),
    ("X-03", "Set up the SES sending identity: verify the domain with DKIM, request production access (leave the SES sandbox), and create the SES SMTP credentials, which go into the SOPS secrets.", "D-12", "W5 staging, go-live", "open", "Needs X-01."),
    ("X-04", "Name the clinical approver and complete the sign-off of D-05, D-06, D-21, D-22, D-28, D-29 (en-US resources) and D-30.", "clinical set", "go-live (W1 builds on the defaults)", "open", ""),
    ("X-05", "Name at least two on-call clinicians (primary and backup, with E.164 phone numbers) and the fallback mailbox.", "D-13", "go-live", "open", "The release-gate rehearsal uses synthetic clinicians."),
    ("X-06", "Legal review of D-23 (consents), D-24 (erasure and retention), D-31 (go-live gate) and the BAA scope for D-04 and D-12.", "legal set", "go-live", "open", ""),
    ("X-07", "Generate one age key pair per environment (staging, production). Keep the private keys off the host, decide who holds them and how they are backed up, and give the controller only the public recipients.", "D-07", "public recipients by the end of W3; private key at the W5 staging deploy", "open", "Tests use throwaway keys. No private key ever goes into the repo, a prompt or a report."),
    ("X-08", "Choose the Bedrock model id(s) for `BEDROCK_MODEL_ID` after the controller lists what is available in us-east-1 (foundation models and inference profiles).", "D-04", "W5 staging deploy", "open", "Never guessed in code or tests."),
    ("X-09", "List any further pilot countries and supply clinician-verified crisis resources for each before its users join.", "D-29", "before the first non-US user", "open", "W1 ships en-US plus the generic fallback."),
    ("X-10", "Decide the commit policy after the program.", "D-17", "after W5", "open", "Nothing is committed until then."),
    ("X-11", "Approve `git config gc.pruneExpire never` in this repo for the program's duration, so checkpoint trees cannot be pruned, or accept the barrier zips as the only protection.", "D-17", "W0 entry (recommended)", "open", "Checkpoint trees are unreferenced git objects; by default git gc may prune them after about two weeks."),
    ("X-12", "Name the off-machine location where barrier zips are copied (another disk or cloud storage).", "D-17", "W0 barrier", "open", ""),
    ("X-13", "Say \"go\" to start W0.", "all", "W0", "open", "Execution starts only on this word."),
]
O = []
o = O.append
o("# MVP open items\n")
o("Owner actions, sign-offs and risks that sit outside the task plans. Update an item's *Status* and *Updated* cells when it changes, and add a dated line to [LOG.md](LOG.md).\n")
o("Status values: `open`, `in progress`, `done`, `n/a`.\n")
o("## Owner actions\n")
o("| ID | Action | Decision | Needed by | Status | Updated | Notes |")
o("|---|---|---|---|---|---|---|")
for x in X:
    o(f"| {x[0]} | {cell(x[1])} | {x[2]} | {x[3]} | `{x[4]}` | 2026-10-19 | {cell(x[5])} |")
o("")
o("## Sign-offs\n")
o("| Area | Covers | Approver | Date | Status | Evidence |")
o("|---|---|---|---|---|---|")
o("| Clinical | D-05 lexicon and thresholds, D-06 crisis watch and re-alerting, D-21 clinician access scope, D-22 screening architecture, D-28 single alert path, D-29 crisis resources per locale, D-30 no diagnosis labels | _(name)_ | | `open` | |")
o("| Legal | D-23 consents, D-24 erasure and retention, D-31 go-live gate, BAA scope for D-04 and D-12 | _(name)_ | | `open` | |")
o("| Go-live gate (D-31) | W5 release gate passed twice; X-02 BAA recorded; clinical sign-off; X-05 on-call named; D-09 restore-drill evidence | owner | | `open` | |")
o("")
o("## Risks\n")
o("Risks from the plan (master plan, section 8), with status. Add new risks at the bottom with the next id.\n")
o("| ID | Risk and mitigation | Status | Updated |")
o("|---|---|---|---|")
for i, r in enumerate(sk["risks"], 1):
    o(f"| R-{i:02d} | {cell(r)} | `open` | 2026-10-19 |")
o("")
OPEN = "\n".join(O) + "\n"

os.makedirs(MVP, exist_ok=True)
for name, body in (("DECISIONS.md", DEC), ("OPEN-ITEMS.md", OPEN)):
    path = os.path.join(MVP, name)
    if os.path.exists(path) and not force:
        print(f"skip {name} (exists; hand-maintained; use --force to regenerate)")
        continue
    open(path, "w", encoding="utf-8", newline="\n").write(body)
    print(f"wrote {path} {len(body)} bytes")
print("impact:", {k: len(v) for k, v in sorted(impact.items())})
