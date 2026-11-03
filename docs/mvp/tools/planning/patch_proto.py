f = r"D:\tmp\mvp-plan\master_protocol.md"
s = open(f, encoding="utf-8-sig").read()
reps = [
    ("no commits, parallel lanes, and AGY as the third model", "no commits, parallel dependency-ready dispatch, and AGY as the third model"),
    ("Any changed path that no running lane declared is a scope breach", "Any changed path that no running task declared is a scope breach"),
    ("a fix task `WS-NN.F<k>` in the owning workstream's lane", "a fix task `WS-NN.F<k>` in the owning workstream, dispatched like any other task"),
]
for a, b in reps:
    if a in s:
        s = s.replace(a, b)
    else:
        print("NOT FOUND:", a[:70])
open(f, "w", encoding="utf-8").write(s)
print("lane mentions left:", [l.strip()[:120] for l in s.splitlines() if "lane" in l.lower()])
