"""Parse workstream plan markdown into task indexes (the format check_plans2.py reads).

usage: parse_plans.py [WS-NN ...]   (default: every WS-*.md in the plan dir)
writes D:\\tmp\\mvp-plan\\indexes_parsed.json  {"indexes": [...]} and requests\\<WS>.md (incoming cross-ws requests)
"""
import json, os, re, sys, collections

REPO = r"D:\Repo\Contextual-Chatbot"
P = r"D:\tmp\mvp-plan"
REL = "docs/superpowers/plans/2026-10-18-mvp-remediation"
DIR = os.path.join(REPO, REL.replace("/", os.sep))
only = set(sys.argv[1:])

TID = r"WS-\d{2}\.[A-Za-z0-9]+"
EXT = re.compile(r"\.(py|md|yml|yaml|json|txt|toml|ini|cfg|sh|ps1|lock|in|sql|conf|j2|tmpl|html|example|pem|crt|key|csv)$", re.I)
BARE = {"Dockerfile", "Caddyfile", "Makefile", ".env.example", ".dockerignore", ".gitignore", ".pre-commit-config.yaml", "alembic.ini", "pytest.ini", "pyproject.toml"}


_skel = json.load(open(os.path.join(P, "skeleton.json"), encoding="utf-8"))
TOP = set(os.listdir(REPO)) | {p.split("/")[0] for w in _skel["workstreams"] for p in w["owned_paths"]} | {
    "deploy", "docker", "scripts", "tests", "docs", "src", "services", "migrations", "infrastructure", ".github"}


ROOTFILE = re.compile(r"(docker-compose[\w.-]*\.ya?ml|requirements[\w.-]*\.(txt|in)|constraints[\w.-]*\.txt|"
                      + "|".join(re.escape(f) for f in sorted(os.listdir(REPO)) if os.path.isfile(os.path.join(REPO, f)) and EXT.search(f)) + ")")


def paths_in(text):
    out = []
    outside = text
    while True:  # drop parenthetical notes, so a root file only mentioned in one is not an edit
        nxt = re.sub(r"\([^()]*\)", "", outside)
        if nxt == outside:
            break
        outside = nxt
    for tok in re.findall(r"`([^`]+)`", text):
        tok = tok.strip()
        if "/" not in tok and ROOTFILE.fullmatch(tok.split("::")[0]) and f"`{tok}`" not in outside:
            continue
        if " " in tok or tok.startswith(("$", "&", "-", "http", "/", "~")) or any(c in tok for c in "<>{}*"):
            continue
        if re.fullmatch(r"[A-Z0-9_/]+", tok.split("::")[0]):
            continue  # env-var groups such as USER_DB_HOST/PORT
        first = tok.replace("\\", "/").split("/")[0]
        if "/" not in tok and ROOTFILE.fullmatch(tok.split("::")[0]):
            out.append(tok.split("::")[0])  # repo-root files such as docker-compose.yml, requirements-runtime.txt
            continue
        if first not in TOP and first not in BARE:
            continue
        tok = tok.split("::")[0]
        tok = re.sub(r":\d+(-\d+)?$", "", tok)
        tok = tok.replace("\\", "/").rstrip("/,;")
        base = tok.split("/")[-1]
        if ("/" in tok and (EXT.search(tok) or base in BARE or tok.endswith("/"))) or base in BARE or (EXT.search(tok) and "/" in tok):
            out.append(tok)
        elif "/" in tok and "." not in base and not tok.startswith("."):
            out.append(tok)  # directory or extension-less file under a path
    return out


def field(block, name):
    m = re.search(r"^\*\*" + re.escape(name) + r":\*\*\s*(.*)$", block, re.M)
    return m.group(1).strip() if m else None


indexes, problems, requests = [], collections.defaultdict(list), collections.defaultdict(list)
for fn in sorted(os.listdir(DIR)):
    m = re.match(r"(WS-\d{2})-.*\.md$", fn)
    if not m:
        continue
    ws = m.group(1)
    if only and ws not in only:
        continue
    text = open(os.path.join(DIR, fn), encoding="utf-8").read()
    heads = list(re.finditer(r"^### Task (" + TID + r"):\s*(.*)$", text, re.M))
    tasks = []
    for i, h in enumerate(heads):
        start = h.end()
        nxt = [x.start() for x in re.finditer(r"^(### Task |## )", text[start:], re.M)]
        block = text[start:start + nxt[0]] if nxt else text[start:]
        tid, name = h.group(1), h.group(2).strip()
        t = {"id": tid, "name": name}
        cov = field(block, "Covers") or ""
        t["covers"] = sorted(set(re.findall(r"MVP-\d{3}", cov)))
        sv = (field(block, "Severity") or "").lower()
        t["severity"] = next((s for s in ("blocker", "high", "medium", "low") if s in sv), "?")
        wv = re.search(r"\d+", field(block, "Wave") or "")
        t["wave"] = int(wv.group()) if wv else -1
        dep = field(block, "Depends on") or ""
        t["depends_on"] = sorted(set(re.findall(TID, dep)) - {tid})
        t["depends_raw"] = dep
        mdl = (field(block, "Implementer model") or "").lower()
        t["model"] = "opus" if "opus" in mdl else ("sonnet" if "sonnet" in mdl else "?")
        rv = (field(block, "Review") or "").lower()
        t["review"] = next((r for r in ("safety-critical", "security-critical", "standard") if r in rv), "?")
        t["decision_gate"] = field(block, "Decision gate") or "none"
        ri = (field(block, "Real infra") or "").lower()
        t["real_infra"] = ri.startswith("yes")
        fm = re.search(r"^\*\*Files:\*\*(.*?)(?=^\*\*Interfaces:\*\*|^- \[ \]|\Z)", block, re.M | re.S)
        cr, md, ts = [], [], []
        cat = None
        if fm:
            for line in fm.group(1).splitlines():
                s = line.strip()
                low = s.lower()
                if re.match(r"^[-*]\s*(create|new)", low):
                    cat = "c"
                elif re.match(r"^[-*]\s*(modify|edit|update|change|delete|remove|rename|move)", low):
                    cat = "m"
                elif re.match(r"^[-*]\s*test", low):
                    cat = "t"
                ps = paths_in(s)
                if not ps:
                    continue
                if cat is None:
                    cat = "m"
                (cr if cat == "c" else md if cat == "m" else ts).extend(ps)
        # tests listed under Create inside tests dirs also count as tests
        t["files_create"] = list(dict.fromkeys(cr))
        t["files_modify"] = list(dict.fromkeys(md))
        t["files_test"] = list(dict.fromkeys(ts + [p for p in cr if "/tests/" in p or p.startswith("tests/")]))
        for k in ("severity", "model", "review"):
            if t[k] == "?":
                problems[ws].append(f"{tid}: could not parse {k}")
        if t["wave"] < 0:
            problems[ws].append(f"{tid}: could not parse wave")
        if not t["covers"]:
            problems[ws].append(f"{tid}: no MVP ids in Covers")
        if not (cr or md):
            problems[ws].append(f"{tid}: no Create/Modify paths parsed")
        tasks.append(t)

    # cross-workstream requests
    xs = []
    sec = re.search(r"^## Cross-workstream requests\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    if sec:
        body = sec.group(1)
        rows = [r for r in body.splitlines() if r.strip().startswith("|")]
        if len(rows) >= 2:
            hdr = [c.strip().lower() for c in rows[0].strip().strip("|").split("|")]
            def col(*names):
                for i, h in enumerate(hdr):
                    if any(n in h for n in names):
                        return i
                return None
            ci_o, ci_c, ci_f, ci_n = col("owner", "to"), col("change", "request", "what"), col("file"), col("needed", "by task", "blocks")
            for r in rows[2:]:
                cells = [c.strip() for c in re.split(r"(?<!\\)\|", r.strip().strip("|"))]
                if len(cells) < len(hdr) - 1:
                    continue
                own = re.findall(r"WS-\d{2}", cells[ci_o]) if ci_o is not None and ci_o < len(cells) else []
                if not own:
                    continue
                change = cells[ci_c] if ci_c is not None and ci_c < len(cells) else r
                if re.search(r"\b(withdrawn|declined by owner|acknowledged, no task)\b", r, re.I):
                    continue
                files = paths_in(cells[ci_f]) if ci_f is not None and ci_f < len(cells) else paths_in(r)
                need = re.findall(TID, cells[ci_n]) if ci_n is not None and ci_n < len(cells) else re.findall(TID, r)
                for o in own:
                    xs.append({"owner_ws": o, "change": change, "files": files, "needed_by_task": need[0] if need else ""})
                    requests[o].append(f"- From {ws} (needed by {', '.join(need) or 'n/a'}): {change} [files: {', '.join(files) or 'n/a'}]")
        else:
            for b in re.split(r"\n(?=[-*] |\d+\. )", body):
                own = re.findall(r"\bWS-\d{2}\b", b)
                own = [o for o in own if o != ws]
                if not own:
                    continue
                need = re.findall(TID, b)
                need = [n for n in need if n.startswith(ws + ".")]
                xs.append({"owner_ws": own[0], "change": " ".join(b.split())[:1500], "files": paths_in(b), "needed_by_task": need[0] if need else ""})
                requests[own[0]].append(f"- From {ws} (needed by {', '.join(need) or 'n/a'}): {' '.join(b.split())[:1500]}")
    else:
        problems[ws].append("no '## Cross-workstream requests' section")
    rf = re.findall(r"REVIEW_FOCUS:\s*[^\n`]*", text)
    indexes.append({"ws": ws, "file": f"{REL}/{fn}", "tasks": tasks, "cross_ws_requests": xs,
                    "new_interfaces": [], "uncovered_ids": [],
                    "notes": "parsed from markdown by parse_plans.py\n" + (rf[-1] if rf else "")})

json.dump({"indexes": indexes}, open(os.path.join(P, "indexes_parsed.json"), "w", encoding="utf-8"), indent=1)
os.makedirs(os.path.join(P, "requests"), exist_ok=True)
for o, lst in requests.items():
    with open(os.path.join(P, "requests", f"{o}.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# Cross-workstream requests addressed to {o} (from the plans written so far)\n\n" + "\n".join(lst) + "\n")

skel = json.load(open(os.path.join(P, "skeleton.json"), encoding="utf-8"))
assigned = {w["id"]: set(w["issue_ids"]) for w in skel["workstreams"]}
for ix in indexes:
    cov = set(m for t in ix["tasks"] for m in t["covers"])
    miss = sorted(assigned[ix["ws"]] - cov)
    print(f"{ix['ws']}: tasks={len(ix['tasks'])} covers {len(cov & assigned[ix['ws']])}/{len(assigned[ix['ws']])} assigned"
          f"{' MISSING ' + ','.join(miss) if miss else ''} | xreq={len(ix['cross_ws_requests'])} | parse problems={len(problems[ix['ws']])}")
    for p in problems[ix["ws"]][:8]:
        print("     ", p)
print("incoming requests per owner:", {o: len(v) for o, v in sorted(requests.items())})
