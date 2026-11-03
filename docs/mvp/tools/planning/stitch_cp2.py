P = r"D:\tmp\mvp-plan"
head = open(P + r"\cp2_head.txt", encoding="utf-8-sig").read()
sched = open(P + r"\cp2_sched.txt", encoding="utf-8").read()
tail = open(P + r"\cp2_tail.txt", encoding="utf-8-sig").read()
old = """    for wv, phs in sorted(schedule.items()):
        for ph, lanes in phs.items():
            fh.write(f"wave {wv} phase {ph}: {sum(len(l) for l in lanes)} tasks in {len(lanes)} lanes; largest lane {max(len(l) for l in lanes)}; "
                     f"real-infra lanes {sum(1 for l in lanes if any(tasks[i]['real_infra'] for i in l))}\\n")"""
new = """    tot = 0
    for wv, phs in sorted(schedule.items()):
        for ph, s in phs.items():
            tot += s["rounds"]
            fh.write(f"wave {wv} phase {ph}: {s['tasks']} tasks; critical path {s['critical_path']}; "
                     f"simulated rounds ({WORKERS} workers, {INFRA} real-infra) {s['rounds']}; max parallel {max(len(b) for b in s['sim'])}\\n")
    fh.write(f"total simulated rounds: {tot} for {len(tasks)} tasks\\n")"""
assert old in tail, "report block not found"
tail = tail.replace(old, new)
out = head + sched + tail
open(P + r"\check_plans2.py", "w", encoding="utf-8").write(out)
import ast
ast.parse(out)
print("stitched OK,", len(out), "chars")
