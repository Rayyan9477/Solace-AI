import json, glob, os, sys
STATE = r"C:\Users\rayyan.a\.claude\plugins\data\gemini-agy-plugin-cc\state"
for line in open(r"D:\tmp\mvp-plan\agy_jobs_round1.tsv", encoding="utf-8-sig"):
    g, jid, wss = line.strip().split("\t")
    hit = glob.glob(os.path.join(STATE, "*", "jobs", f"{jid}.json"))[0]
    rec = json.load(open(hit, encoding="utf-8"))
    res = rec.get("result") or {}
    print("==", g, wss, "keys:", list(rec.keys()), "result keys:", list(res.keys()))
    print("   raw:", repr((res.get("rawOutput") or "")[:400]))
    for k in ("output", "text", "message", "stdout"):
        if res.get(k):
            print(f"   {k}:", repr(str(res[k])[:300]))
    log = hit[:-5] + ".log"
    if os.path.exists(log):
        t = open(log, encoding="utf-8", errors="replace").read()
        print("   log tail:", repr(t[-700:]))
    # any other files for this job
    others = [p for p in glob.glob(os.path.join(os.path.dirname(hit), jid + "*")) if not p.endswith((".json", ".log"))]
    print("   other files:", others)
