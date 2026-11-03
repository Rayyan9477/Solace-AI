import json
sk = json.load(open(r"D:\tmp\mvp-plan\skeleton.json", encoding="utf-8"))
for k in ("global_constraints", "review_focus", "risks"):
    v = sk[k]
    print(f"== {k}: type={type(v).__name__} n={len(v)}")
    for x in v[:3]:
        print("   ", json.dumps(x)[:400])
print("== test_strategy type:", type(sk["test_strategy"]).__name__, "len:", len(json.dumps(sk["test_strategy"])))
print(json.dumps(sk["test_strategy"])[:800])
print("== wave keys:", list(sk["waves"][0].keys()))
w0 = sk["waves"][0]
for k in ("entry_criteria", "exit_criteria", "bug_round_focus"):
    print(f"   {k}: type={type(w0[k]).__name__}", json.dumps(w0[k])[:300])
print("== interface keys:", list(sk["interfaces"][0].keys()), "contract len:", len(sk["interfaces"][0]["contract"]))
print("== decision keys:", list(sk["decisions"][0].keys()))
print("== shared_files keys:", list(sk["shared_files"][0].keys()))
for wid in ("WS-05", "WS-10", "WS-11", "WS-12", "WS-17"):
    w = next(x for x in sk["workstreams"] if x["id"] == wid)
    print(f"== {wid} owned_paths:", w["owned_paths"])
print("== ws keys:", list(sk["workstreams"][0].keys()))
print("== total json chars:", len(json.dumps(sk)))
