"""Prepare inputs for the compile step (AGY scratch workspace + re-grade candidates)."""
import json, os, re, shutil

ST = r"D:\tmp\mvp-review\state"
OUT = r"D:\tmp\mvp-plan"
WS = os.path.join(OUT, "agy-ws")
REPO = r"D:\Repo\Contextual-Chatbot"
os.makedirs(WS, exist_ok=True)

reg = json.load(open(os.path.join(ST, "register.json"), encoding="utf-8"))
syn = json.load(open(os.path.join(ST, "synth.json"), encoding="utf-8"))
by = {r["mid"]: r for r in reg}
cluster_of = {}
for c in syn["clusters"]:
    for i in c["ids"]:
        cluster_of[i] = c["name"]


def clip(s, n):
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def row(r, n_ev=260, n_sc=260):
    return (f"{r['mid']} | {r['sev']} | {cluster_of.get(r['mid'], '?')} | {r['area']} | {r['cat']} | "
            f"{clip(r['title'], 260)} | evidence: {clip(r['evidence'], n_ev)} | scenario: {clip(r['scenario'], n_sc)} | "
            f"sources: {', '.join(r['ids'])}")


with open(os.path.join(WS, "register.md"), "w", encoding="utf-8") as fh:
    fh.write("# Verified MVP register (344 entries)\n\nFormat: id | severity | cluster | area | category | title | evidence | scenario | sources\n\n")
    for r in reg:
        fh.write("- " + row(r, 200, 200) + "\n")

with open(os.path.join(WS, "overlaps.md"), "w", encoding="utf-8") as fh:
    fh.write("# Possible overlaps to resolve\n\n")
    for k, o in enumerate(syn.get("possible_overlaps") or [], 1):
        fh.write(f"## Group O-{k:02d}: {', '.join(o['ids'])}\nSynthesizer note: {o.get('note')}\n")
        for i in o["ids"]:
            if i in by:
                fh.write("- " + row(by[i], 600, 600) + "\n")
        fh.write("\n")

for name in ["ENTERPRISE-READINESS-ASSESSMENT.md", "FUNCTIONAL-COMPLETENESS.md", "BUG-BACKLOG.md",
             "PHASE-C-SECURITY-FINDINGS.md", "KNOWN-LIMITATIONS.md", "POST-MVP-BACKLOG.md",
             "CLINICAL-VALIDATION.md", "ENTERPRISE-RELEASE-PLAN.md"]:
    shutil.copy(os.path.join(REPO, "docs", name), os.path.join(WS, name))
shutil.copy(r"D:\tmp\solace-full-picture-report.md", os.path.join(WS, "FULL-PICTURE-REPORT-2026-10-11.md"))

# scope-sensitive candidates for re-grading under "Pilot + enterprise basics"
pat = re.compile(r"\b(backup|restore|pitr|pg_dump|wal|snapshot|disaster|mfa|2fa|totp|otp|second factor|multi-factor|"
                 r"audit|tamper|hmac|secret|secrets|vault|key rotation|rotate|master key|encryption_key|jwt_secret|"
                 r"plaintext env|\.env|kms|credential)\b", re.I)
cands = [r for r in reg if r["sev"] == "post-mvp" or pat.search(r["title"] + " " + r["scenario"] + " " + r["evidence"])]
with open(os.path.join(OUT, "regrade_candidates.md"), "w", encoding="utf-8") as fh:
    for r in cands:
        fh.write("- " + row(r, 400, 400) + "\n")
print("register rows:", len(reg), "overlap groups:", len(syn.get("possible_overlaps") or []), "regrade candidates:", len(cands))
print({s: sum(1 for r in cands if r["sev"] == s) for s in ["blocker", "high", "medium", "low", "post-mvp"]})
print(sorted(os.listdir(WS)))
