import re
src = open(r"D:\tmp\mvp-plan\wf_skeleton.js", encoding="utf-8").read().splitlines()
for i, line in enumerate(src, 1):
    if line.startswith("const CTX"):
        print("line", i, "len", len(line))
        print(line[:600])
        print("...")
        print(line[-400:])
        print("template refs:", re.findall(r"\$\{[^}]+\}", line))
