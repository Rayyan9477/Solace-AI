"""The issue register must not call an entry fixed while a task it names for that fix is still open (W0-close
review DW-DOC-1: section 4.3 said MVP-364 to MVP-366 were fixed by WS-04.10, WS-05.8 and WS-13.8, all todo)."""
from __future__ import annotations

import importlib.util
import re
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTER = ROOT / "docs" / "MVP-ISSUE-REGISTER.md"
TASK_ID = re.compile(r"WS-\d{2}\.[A-Za-z0-9]+")


def _tracker_status() -> dict[str, str]:
    spec = importlib.util.spec_from_file_location("mvp_docs", ROOT / "docs" / "mvp" / "tools" / "mvp_docs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {task: row["status"] for task, row in module.read_rows().items()}


def _added_entries() -> Iterator[tuple[str, str]]:
    """(id, Why cell) for every row of section 4.3, Added entries."""
    section = REGISTER.read_text(encoding="utf-8").split("### 4.3 Added entries", 1)[1].split("\n## ", 1)[0]
    for line in section.splitlines():
        m = re.match(r"^\| (MVP-\d{3}) \|", line)
        if m:
            yield m.group(1), re.split(r"(?<!\\)\|", line.strip().strip("|"))[-1].strip()


def _fixed_claims(why: str) -> list[str]:
    """Task ids named in a sentence that says the entry is fixed (a 'partly fixed' sentence is not a claim)."""
    return [task for sentence in re.split(r"(?<=\.)\s+", why)
            if re.search(r"(?<!partly )\bfixed (?:in|by)\b", sentence, re.I) for task in TASK_ID.findall(sentence)]


def test_the_claim_check_reads_the_register():
    claims = {entry: _fixed_claims(why) for entry, why in _added_entries()}
    assert len(claims) >= 22
    assert claims["MVP-361"] == ["WS-03.F1"]
    assert _fixed_claims("Partly fixed in the W0 close lane (WS-04.F2). The remainder is owned by WS-04.10.") == []
    assert _fixed_claims("Found by the review and partly fixed by WS-04.F2. Evidence: x") == []


def test_register_fix_claims_name_only_done_tasks():
    status = _tracker_status()
    open_claims = {entry: [(task, status.get(task, "not in TRACKER")) for task in _fixed_claims(why)
                           if status.get(task) != "done"]
                   for entry, why in _added_entries()}
    open_claims = {entry: tasks for entry, tasks in open_claims.items() if tasks}
    assert open_claims == {}, f"register entries said to be fixed by tasks that are not done: {open_claims}"
