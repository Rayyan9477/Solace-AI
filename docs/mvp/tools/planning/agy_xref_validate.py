import json, re, collections

p = r"C:\Users\rayyan.a\.claude\plugins\data\gemini-agy-plugin-cc\state\agy-ws-b89020f977f92bf7\jobs\task-mvcv78cx-3a599fcbeb.json"
rec = json.load(open(p, encoding="utf-8"))
res = rec["result"]
print("failure:", json.dumps(res.get("failure"))[:500])
raw = res["rawOutput"]
block = re.findall(r"```json\s*(.*?)```", raw, re.S)[-1].strip()
items, end = json.JSONDecoder().raw_decode(block)
print("items:", len(items))

reg = json.load(open(r"D:\tmp\mvp-review\state\register.json", encoding="utf-8"))
valid = {r["mid"] for r in reg}
bad_ids = [(it["tracker"], it["id"], m) for it in items for m in it.get("mvp_ids") or [] if m not in valid]
print("cited ids not in register:", bad_ids[:10], len(bad_ids))

keys = collections.Counter((it["tracker"], it["id"]) for it in items)
print("duplicate items:", [k for k, v in keys.items() if v > 1])
er = {it["id"] for it in items if it["tracker"] == "ENTERPRISE-READINESS"}
fp = {it["id"] for it in items if it["tracker"] == "FULL-PICTURE"}
need_blk = {f"BLK-{i:02d}" for i in range(1, 34)}
need_h = {f"H-{i:02d}" for i in range(1, 24)}
need_fp = {f"FP-N-{i}" for i in range(1, 29)}
print("missing BLK:", sorted(need_blk - er), "missing H:", sorted(need_h - er), "missing FP:", sorted(need_fp - fp, key=lambda x: int(x.split('-')[-1])))
print("by tracker:", collections.Counter(it["tracker"] for it in items))
print("coverage:", collections.Counter(it["coverage"] for it in items))
print("status:", collections.Counter(it["status_in_tracker"] for it in items))
gaps = [it for it in items if it["coverage"] in ("none", "partial") and it["status_in_tracker"] in ("open", "partial", "unknown")]
print("open items with coverage none/partial:", len(gaps))
for it in gaps:
    print(f"  {it['coverage']:7} {it['tracker']}:{it['id']} | {it['title'][:90]} | {it.get('note','')[:160]}")
json.dump(items, open(r"D:\tmp\mvp-plan\agy_crossref.json", "w", encoding="utf-8"), indent=1)
json.dump(gaps, open(r"D:\tmp\mvp-plan\xref_gaps.json", "w", encoding="utf-8"), indent=1)
