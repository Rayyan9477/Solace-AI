"""Extract the fenced JSON block from a stored AGY job result into a file.

usage: agy_extract.py <job-id> <out.json>
"""
import json, os, re, sys

JOBS = r"C:\Users\rayyan.a\.claude\plugins\data\gemini-agy-plugin-cc\state\agy-ws-b89020f977f92bf7\jobs"
job_id, out = sys.argv[1], sys.argv[2]
rec = json.load(open(os.path.join(JOBS, job_id + ".json"), encoding="utf-8"))


def find_raw(o):
    if isinstance(o, dict):
        if isinstance(o.get("rawOutput"), str):
            return o["rawOutput"]
        for v in o.values():
            r = find_raw(v)
            if r:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find_raw(v)
            if r:
                return r
    return None


raw = find_raw(rec)
if not raw:
    sys.exit(f"no rawOutput in job {job_id} (status {rec.get('status')})")
blocks = re.findall(r"```json\s*(.*?)```", raw, re.S)
text = blocks[-1] if blocks else raw
data = json.loads(text)
json.dump(data, open(out, "w", encoding="utf-8"), indent=1)
print(f"{job_id}: status={rec.get('status')} items={len(data)} -> {out}")
