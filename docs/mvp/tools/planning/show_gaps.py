import json

g = json.load(open(r"D:\tmp\mvp-plan\gaps_verified.json", encoding="utf-8"))
src = [f"{x['tracker']}:{x['id']}" for x in json.load(open(r"D:\tmp\mvp-plan\xref_gaps.json", encoding="utf-8"))]
ids = [e["tracker_id"] for e in g["entries"]] + [r["tracker_id"] for r in g["rejected"]]
print("all 24 accounted:", sorted(ids) == sorted(src), len(ids), "dups:", len(ids) - len(set(ids)))
print("ADDED:")
for e in g["entries"]:
    print(f"  {e['severity']:8} | {e['tracker_id']} | {e['area']} | {e['cluster'][:40]} | {e['title'][:150]}")
    print(f"           ev: {e['evidence'][:200]}")
print("REJECTED:")
for r in g["rejected"]:
    print(f"  {r['tracker_id']} | {r['reason'][:170]}")
