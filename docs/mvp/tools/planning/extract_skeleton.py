import json, collections, itertools

OUT = r"C:\Users\rayyan.a\AppData\Local\Temp\claude\d--Repo-Contextual-Chatbot\f4d89165-f54d-4922-a3bc-d6f5a6eee40c\tasks\w57gcvet4.output"
obj = json.load(open(OUT, encoding="utf-8"))
res = obj.get("result", obj) if isinstance(obj, dict) else obj
if isinstance(res, str):
    res = json.loads(res)
sk = res["skeleton"]
json.dump(sk, open(r"D:\tmp\mvp-plan\skeleton.json", "w", encoding="utf-8"), indent=1)
print("top keys:", list(res.keys()))
print("remainingProblems:", json.dumps(res.get("remainingProblems"))[:3000])
print("skeleton keys:", list(sk.keys()))
print("summary chars:", len(sk["summary"]))

reg = json.load(open(r"D:\tmp\mvp-plan\register_compiled.json", encoding="utf-8"))
entries = reg["entries"] if isinstance(reg, dict) else reg
required = {e["mid"] for e in entries if e["sev"] != "post-mvp"}
backlog = {e["mid"] for e in entries if e["sev"] == "post-mvp"}
sev = {e["mid"]: e["sev"] for e in entries}
print("required:", len(required), "backlog:", len(backlog))

cnt = collections.Counter()
print("\nWORKSTREAMS")
for w in sk["workstreams"]:
    cnt.update(w["issue_ids"])
    s = collections.Counter(sev.get(i, "?") for i in w["issue_ids"])
    deps = ",".join(d["ws"] for d in w.get("depends_on", []))
    print(f"{w['id']} {w['slug']:28} waves={w['waves']} n={len(w['issue_ids'])} "
          f"B{s['blocker']} H{s['high']} M{s['medium']} L{s['low']} ?{s['?']} paths={len(w['owned_paths'])} risk={w['risk']} deps=[{deps}]")
    print(f"     {w['name']}")
allids = set(cnt)
print("\nmissing:", sorted(required - allids))
print("dups:", sorted(k for k, v in cnt.items() if v > 1))
print("unknown:", sorted(allids - required - backlog))
print("backlog assigned:", sorted(allids & backlog))

print("\nWAVES")
for wv in sk["waves"]:
    print(f"W{wv['wave']} {wv['name']} | ws={wv['workstreams']}")
    print(f"     goal: {wv['goal'][:300]}")

print("\nINTERFACES")
for i in sk["interfaces"]:
    print(f"{i['id']} {i['name'][:70]} owner={i['owner_ws']} consumers={i.get('consumers')}")

print("\nDECISIONS")
for d in sk["decisions"]:
    print(f"{d['id']} W{d.get('needed_by_wave')} owner={d.get('owner')} | {d['question'][:160]}")
    print(f"     default: {d['recommended_default'][:200]}")

print("\nshared_files:", len(sk.get("shared_files", [])))
for s in sk.get("shared_files", [])[:40]:
    print("  ", json.dumps(s)[:220])
print("global_constraints:", len(sk["global_constraints"]), "review_focus:", len(sk["review_focus"]), "risks:", len(sk.get("risks", [])))

# path overlap check
def overlaps(a, b):
    a2, b2 = a.rstrip("/"), b.rstrip("/")
    return a2 == b2 or a2.startswith(b2 + "/") or b2.startswith(a2 + "/")
shared_paths = set()
for s in sk.get("shared_files", []):
    if isinstance(s, dict):
        shared_paths.add(s.get("path", ""))
    else:
        shared_paths.add(str(s))
print("\nOVERLAPS (not declared shared):")
ws = sk["workstreams"]
n = 0
for x, y in itertools.combinations(ws, 2):
    for a in x["owned_paths"]:
        for b in y["owned_paths"]:
            if overlaps(a, b) and a not in shared_paths and b not in shared_paths:
                n += 1
                if n <= 60:
                    print(f"  {x['id']}:{a}  <->  {y['id']}:{b}")
print("overlap count:", n)
