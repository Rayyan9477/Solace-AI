f = r"D:\tmp\mvp-plan\assemble_master.py"
s = open(f, encoding="utf-8").read()
reps = [
    ("(no git commits, parallel file-disjoint lanes, AGY as a third reviewer)", "(no git commits, parallel dispatch of file-disjoint tasks, AGY as a third reviewer)"),
    ("executed as parallel lanes in one working tree with commit-free checkpoints", "executed in parallel in one working tree with commit-free checkpoints"),
    ("Inside a wave, run phase A lanes (platform) before phase B lanes (services), and run each lane's tasks in the listed order.",
     "Inside a wave, run phase A tasks (platform) before phase B tasks (services), and dispatch each task when its dependencies are done and its files and suites are free (section 5.3)."),
]
for a, b in reps:
    if a in s:
        s = s.replace(a, b)
    else:
        print("NOT FOUND:", a[:80])
open(f, "w", encoding="utf-8").write(s)
import ast
ast.parse(s)
print("lane mentions left:", [l.strip()[:120] for l in s.splitlines() if "lane" in l.lower()])
