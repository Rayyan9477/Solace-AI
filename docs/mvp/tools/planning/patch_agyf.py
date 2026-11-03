f = r"D:\tmp\mvp-plan\agy_findings.py"
s = open(f, encoding="utf-8").read()
reps = [
    ('json.dump(all_f, open(os.path.join(P, "agy_findings.json"), "w", encoding="utf-8"), indent=1)',
     'json.dump(all_f, open(os.path.join(P, OUTNAME + ".json"), "w", encoding="utf-8"), indent=1)'),
    ('os.makedirs(os.path.join(P, "fix"), exist_ok=True)', 'os.makedirs(os.path.join(P, FIXDIR), exist_ok=True)'),
    ('path = os.path.join(P, "fix", f"{ws}.md")', 'path = os.path.join(P, FIXDIR, f"{ws}.md")'),
]
for a, b in reps:
    assert a in s, a
    s = s.replace(a, b)
open(f, "w", encoding="utf-8").write(s)
import ast; ast.parse(s); print("ok")
