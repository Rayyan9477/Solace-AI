import json, fnmatch, os, re, sys

src = open(r"D:\tmp\mvp-plan\check_plans2.py", encoding="utf-8").read()
# pull only the resolver section into this namespace
start = src.index("def norm(f):")
end = src.index("# ---------- per-workstream checks ----------")
skel = json.load(open(r"D:\tmp\mvp-plan\skeleton.json", encoding="utf-8"))
ns = {"skel": skel, "fnmatch": fnmatch, "os": os}
exec(src[start:end], ns)
samples = [
    "services/memory_service/src/infrastructure/event_bridge.py",
    "services/therapy_service/src/infrastructure/event_bridge.py",
    "migrations/versions/018_memory.py",
    "migrations/versions/009_x.py",
    "migrations/env.py",
    "services/user-service/src/api_auth.py",
    "services/user-service/src/api.py",
    "services/user-service/src/main.py",
    "services/memory_service/src/main.py",
    "services/memory_service/Dockerfile",
    "services/orchestrator_service/src/main.py",
    "services/orchestrator_service/src/langgraph/graph_builder.py",
    "src/solace_infrastructure/runtime/bootstrap.py",
    "src/solace_infrastructure/__init__.py",
    "src/solace_security/middleware.py",
    "docker-compose.yml",
    ".env.example",
    "tests/realinfra/ws-12/test_x.py",
    "services/safety_service/tests/test_x.py",
]
for s in samples:
    c = ns["owners"](s)
    print(f"{s:65} -> {c[0][1] if c else None:6} via {c[0][4] if c else '-'}   (cands: {[x[1] for x in c]})  platform={ns['is_platform'](s)} suite={ns['suite_root'](s)}")
