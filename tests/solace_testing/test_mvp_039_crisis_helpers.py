"""Unit tests for the episode-scoped crisis helpers of solace_testing.e2e (MVP-039, WS-05.6; crisis-review F6).

No stack: a fake Stack records ``sql`` calls and a fake mail sink keeps SmtpSink's ``messages`` semantics (address
match, case-insensitive; subject substring; oldest first). Synthetic data only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from solace_testing import e2e
from solace_testing.e2e import crisis_episodes, pages_for_episode, wait_for_episode_page
from solace_testing.realinfra import MailMessage

ALERT_A = "0f1e2d3c-aaaa-4bbb-8ccc-000000000001"
ALERT_B = "0f1e2d3d-aaaa-4bbb-8ccc-000000000002"
CLIN = "clin-000000000001@test.invalid"
OTHER = "clin-000000000002@test.invalid"


class FakeMail:
    def __init__(self, messages: list[MailMessage] | None = None) -> None:
        self.held: list[MailMessage] = list(messages or [])
        self.calls = 0

    def messages(self, to: str | None = None, subject_contains: str = "") -> list[MailMessage]:
        self.calls += 1
        return [m for m in self.held
                if (to is None or to.lower() in {a.lower() for a in m.to}) and subject_contains in m.subject]


@dataclass
class FakeStack:
    responses: list[list[dict]] = field(default_factory=list)
    mail: FakeMail = field(default_factory=FakeMail)
    queries: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)

    def sql(self, query: str, *params: str) -> list[dict]:
        self.queries.append((query, params))
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0] if self.responses else []


def _page(to: str, alert_id: str, level: str = "HIGH", text: str = "Open the link to acknowledge.") -> MailMessage:
    return MailMessage(id=uuid4().hex, to=[to], subject=f"[Solace CRISIS {level}] action required - ref {alert_id[:8]}",
                       text=text, headers={})


def _row(alert_id: str, ord_: int, status: str = "open") -> dict:
    return {"alert_id": alert_id, "status": status, "max_level": "HIGH", "sources": ["orchestrator"],
            "page_count": 1, "ord": ord_}


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    clock = [0.0]

    def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(e2e.time, "sleep", fake_sleep)
    monkeypatch.setattr(e2e.time, "monotonic", lambda: clock[0])
    return slept


# --------------------------------------------------------------------------------------------- crisis_episodes


def test_mvp_039_crisis_episodes_queries_the_users_crisis_alerts_with_a_bound_parameter() -> None:
    user_id = uuid4()
    stack = FakeStack(responses=[[_row(ALERT_A, 1)]])
    crisis_episodes(stack, user_id)
    (query, params), = stack.queries
    assert params == (str(user_id),)
    assert str(user_id) not in query                                     # bound, never interpolated
    flat = " ".join(query.split())
    assert "FROM crisis_alerts" in flat and "user_id = :'p1'::uuid" in flat
    for column in ("alert_id::text AS alert_id", "status", "max_level", "sources", "page_count"):
        assert column in flat
    assert "ORDER BY opened_at" in flat


def test_mvp_039_crisis_episodes_returns_the_five_columns_ordered_by_opened_at() -> None:
    stack = FakeStack(responses=[[_row(ALERT_B, 2, "acknowledged"), _row(ALERT_A, 1)]])   # json_agg order is not promised
    rows = crisis_episodes(stack, uuid4())
    assert [r["alert_id"] for r in rows] == [ALERT_A, ALERT_B]
    assert all(set(r) == {"alert_id", "status", "max_level", "sources", "page_count"} for r in rows)
    assert rows[1]["status"] == "acknowledged"


def test_mvp_039_crisis_episodes_without_timeout_queries_once(no_sleep: list[float]) -> None:
    stack = FakeStack(responses=[[]])
    assert crisis_episodes(stack, uuid4()) == []
    assert len(stack.queries) == 1 and no_sleep == []


def test_mvp_039_crisis_episodes_polls_until_a_row_appears(no_sleep: list[float]) -> None:
    stack = FakeStack(responses=[[], [], [_row(ALERT_A, 1)]])
    rows = crisis_episodes(stack, uuid4(), timeout=60)
    assert [r["alert_id"] for r in rows] == [ALERT_A]
    assert len(stack.queries) == 3 and len(no_sleep) == 2


def test_mvp_039_crisis_episodes_gives_up_empty_after_the_timeout(no_sleep: list[float]) -> None:
    stack = FakeStack(responses=[[]])
    assert crisis_episodes(stack, uuid4(), timeout=5) == []
    assert sum(no_sleep) <= 5 + 1.0 and len(stack.queries) >= 2


def test_mvp_039_crisis_episodes_does_not_swallow_sql_errors(no_sleep: list[float]) -> None:
    class Broken(FakeStack):
        def sql(self, query: str, *params: str) -> list[dict]:
            raise RuntimeError('psql exited 1:\nERROR:  relation "crisis_alerts" does not exist')

    with pytest.raises(RuntimeError, match="crisis_alerts"):
        crisis_episodes(Broken(), uuid4(), timeout=30)
    assert no_sleep == []


def test_mvp_039_crisis_episodes_refuses_a_non_uuid_user() -> None:
    with pytest.raises(ValueError):
        crisis_episodes(FakeStack(), "x' OR true --")


# --------------------------------------------------------------------------------------------- pages_for_episode


def test_mvp_039_pages_for_episode_keeps_only_this_episodes_pages_to_this_address() -> None:
    mine1, mine2 = _page(CLIN, ALERT_A, "HIGH"), _page(CLIN, ALERT_A, "CRITICAL")
    stack = FakeStack(mail=FakeMail([mine1, _page(CLIN, ALERT_B), _page(OTHER, ALERT_A), mine2,
                                     MailMessage(id="x", to=[CLIN], subject="Welcome", text="", headers={})]))
    assert pages_for_episode(stack, CLIN, ALERT_A) == [mine1, mine2]
    assert pages_for_episode(stack, CLIN, ALERT_B) != []
    assert pages_for_episode(stack, "nobody@test.invalid", ALERT_A) == []


def test_mvp_039_pages_for_episode_accepts_a_uuid_or_uppercase_id() -> None:
    mine = _page(CLIN, ALERT_A)
    stack = FakeStack(mail=FakeMail([mine]))
    assert pages_for_episode(stack, CLIN, UUID(ALERT_A)) == [mine]
    assert pages_for_episode(stack, CLIN, ALERT_A.upper()) == [mine]


def test_mvp_039_pages_for_episode_refuses_a_non_uuid_alert_id() -> None:
    with pytest.raises(ValueError):
        pages_for_episode(FakeStack(), CLIN, "")


# --------------------------------------------------------------------------------------------- wait_for_episode_page


def test_mvp_039_wait_for_episode_page_returns_the_oldest_page_once_it_arrives(no_sleep: list[float]) -> None:
    first, later = _page(CLIN, ALERT_A, "HIGH"), _page(CLIN, ALERT_A, "CRITICAL")
    mail = FakeMail([_page(CLIN, ALERT_B)])
    stack = FakeStack(mail=mail)
    real_messages = mail.messages

    def arriving(to: str | None = None, subject_contains: str = "") -> list[MailMessage]:
        if mail.calls == 2:
            mail.held += [first, later]
        return real_messages(to, subject_contains)

    mail.messages = arriving  # type: ignore[method-assign]
    assert wait_for_episode_page(stack, CLIN, ALERT_A, timeout=60) is first
    assert len(no_sleep) >= 1


def test_mvp_039_wait_for_episode_page_times_out_without_echoing_mail(no_sleep: list[float]) -> None:
    other = _page(CLIN, ALERT_B, text="I don’t want to be alive anymore")   # another user's episode
    stack = FakeStack(mail=FakeMail([other]))
    with pytest.raises(TimeoutError) as info:
        wait_for_episode_page(stack, CLIN, ALERT_A, timeout=10)
    message = str(info.value)
    assert f"ref {ALERT_A[:8]}" in message and "10" in message
    assert "alive anymore" not in message and other.subject not in message
    assert sum(no_sleep) <= 10 + 1.0


def test_mvp_039_crisis_helpers_are_exported() -> None:
    assert {"crisis_episodes", "pages_for_episode", "wait_for_episode_page"} <= set(e2e.__all__)


# --------------------------------------------------------------------------------------------- suite structure (fix 1)

import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from tests.e2e import test_crisis_alert_e2e as crisis_suite  # noqa: E402
from tests.e2e import test_w0_boot_smoke as boot_suite  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ACK_URL = "http://127.0.0.1:8080/api/v1/notifications/crisis-alerts/ack/tok-synthetic"


def _marks(module, name: str) -> list:
    return [m for m in module.pytestmark if m.name == name]


@pytest.mark.parametrize("module", [crisis_suite, boot_suite], ids=["crisis", "boot_smoke"])
def test_mvp_039_suite_timeouts_bound_the_function_only(module) -> None:
    (timeout,) = _marks(module, "timeout")
    assert timeout.kwargs.get("func_only") is True          # W056-CRI-2: the stack build (UP_TIMEOUT_S) is unbounded


def test_mvp_039_crisis_suite_is_dev_only() -> None:
    (profile,) = _marks(crisis_suite, "profile")
    assert profile.args == ("dev",)                          # W056-CRI-3


def _collect(profile: str) -> str:
    env = {**os.environ, "SOLACE_E2E_PROFILE": profile, "PYTHONPATH": f"{ROOT}{os.pathsep}{ROOT / 'src'}"}
    proc = subprocess.run([sys.executable, "-m", "pytest", "tests/e2e/test_crisis_alert_e2e.py", "--collect-only",
                           "-q", "-o", "addopts=", "-p", "no:logging"], cwd=ROOT, env=env, capture_output=True,
                          text=True, timeout=120)
    return proc.stdout


def test_mvp_039_strict_deselects_the_crisis_suite() -> None:
    assert "8 tests collected" in _collect("dev")
    assert "8 deselected" in _collect("strict")


class _Resp:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code, self.text = status_code, text


class AckStack(FakeStack):
    """GET returns ``get_body``; POST acknowledges. ``get_flips`` makes the GET itself acknowledge (an IF-09 bug)."""

    def __init__(self, get_body: str, *, get_flips: bool = False) -> None:
        super().__init__()
        self.status, self.get_body, self.get_flips, self.calls = "open", get_body, get_flips, []

    async def edge(self, method: str, path: str, **_: object) -> _Resp:
        self.calls.append((method, path))
        if method == "GET":
            if self.get_flips:
                self.status = "acknowledged"
            return _Resp(200, self.get_body)
        self.status = "acknowledged"
        return _Resp(303)

    def sql(self, query: str, *params: str) -> list[dict]:
        return [{**_row(ALERT_A, 1), "status": self.status}]


CONFIRM_FORM = '<html><FORM Method="POST" action=""><button>Acknowledge</button></FORM></html>'


def _ack_msg() -> MailMessage:
    return _page(CLIN, ALERT_A, text=f"Acknowledge: {ACK_URL}")


async def test_mvp_039_ack_hop_passes_for_a_confirm_form_then_post() -> None:
    stack = AckStack(CONFIRM_FORM)
    await crisis_suite._acknowledge(stack, uuid4(), _ack_msg())
    assert [m for m, _ in stack.calls] == ["GET", "POST"] and stack.status == "acknowledged"
    assert stack.calls[0][1] == "/api/v1/notifications/crisis-alerts/ack/tok-synthetic"


async def test_mvp_039_ack_hop_fails_when_the_get_changes_state() -> None:
    with pytest.raises(AssertionError):
        await crisis_suite._acknowledge(AckStack(CONFIRM_FORM, get_flips=True), uuid4(), _ack_msg())


async def test_mvp_039_ack_hop_fails_without_a_post_confirm_form() -> None:
    for body in ("<p>Acknowledged</p>", '<form method="get"><button>ok</button></form>'):
        with pytest.raises(AssertionError):
            await crisis_suite._acknowledge(AckStack(body), uuid4(), _ack_msg())


@pytest.mark.parametrize(("subject_level", "max_level", "ok"), [
    ("HIGH", "HIGH", True), ("CRITICAL", "CRITICAL", True), ("HIGH", "CRITICAL", True), ("CRITICAL", "HIGH", False)])
def test_mvp_039_page_level_is_tied_to_the_episode(subject_level: str, max_level: str, ok: bool) -> None:
    subject = f"[Solace CRISIS {subject_level}] action required - ref {ALERT_A[:8]}"
    if ok:
        crisis_suite._assert_page_level(subject, max_level)
    else:
        with pytest.raises(AssertionError):
            crisis_suite._assert_page_level(subject, max_level)
