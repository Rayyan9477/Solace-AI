import json, glob, os, sys, time
STATE = r"C:\Users\rayyan.a\.claude\plugins\data\gemini-agy-plugin-cc\state"
tsv = sys.argv[1] if len(sys.argv) > 1 else r"D:\tmp\mvp-plan\agy_jobs_round1.tsv"
for line in open(tsv, encoding="utf-8-sig"):
    parts = line.strip().split("\t")
    if len(parts) < 2:
        continue
    g, jid = parts[0], parts[1]
    hits = glob.glob(os.path.join(STATE, "*", "jobs", f"{jid}.json"))
    if not hits:
        print(g, jid, "no record yet")
        continue
    rec = json.load(open(hits[0], encoding="utf-8"))
    res = rec.get("result") or {}
    raw = res.get("rawOutput") or ""
    log = hits[0][:-5] + ".log"
    lsz = os.path.getsize(log) if os.path.exists(log) else 0
    age = int(time.time() - os.path.getmtime(log)) if os.path.exists(log) else -1
    print(f"{g} {jid} status={rec.get('status')} phase={rec.get('phase')} raw={len(raw)} log={lsz}B (updated {age}s ago) fail={json.dumps(res.get('failure'))[:160] if res.get('failure') else ''}")
