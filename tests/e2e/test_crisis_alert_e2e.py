"""Crisis alert E2E, written red at documented hops (MVP-039, WS-05.6; dev profile).

The crisis path end to end: the IF-12 REST ``crisis`` object and WS ``safety.alert`` frame with the IF-07 en-US
resources (D-29), the IF-09 page to the on-call clinician (one per level, no message text, no user email) and the
acknowledgement link that moves the episode (``crisis_alerts.status``) to ``acknowledged``.

Isolation (crisis-review F1/F6): the stack, roster and fallback mailbox are shared by the session, so this module
provisions its on-call clinician before its first crisis message and scopes every page assertion to one user's
episode (``crisis_episodes`` + ``pages_for_episode``), never to a whole mailbox. W0 red hops are recorded in
tests/e2e/expected_failures.yaml (target wave 1, dev).
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
import websockets

from solace_testing.e2e import (
    crisis_episodes,
    extract_link,
    make_user,
    new_email,
    pages_for_episode,
    provision_clinician,
    wait_for_episode_page,
)
from solace_testing.e2e_manifest import hop
from tests.e2e.crisis_phrases_e2e import CRISIS_PHRASES

# func_only: the test function is bounded, the session stack build (UP_TIMEOUT_S) is not (W056-CRI-2)
pytestmark = [pytest.mark.e2e, pytest.mark.integration, pytest.mark.timeout(900, func_only=True),
              pytest.mark.profile("dev")]   # W0/W1 entries are dev-only; strict widening is a W1 item (W056-CRI-3)

LEVELS = ("HIGH", "CRITICAL")
_SUBJECT_RE = re.compile(r"\[Solace CRISIS (HIGH|CRITICAL)\] action required - ref [0-9a-f]{8}")
_POST_FORM_RE = re.compile(r"<form\b[^>]*\bmethod\s*=\s*[\"']?post\b", re.IGNORECASE)


@pytest.fixture(scope="module", autouse=True)
def oncall(e2e_stack):
    # crisis-review F1/F6: the module's on-call clinician exists before its first crisis message, so no page of this
    # module waits on an empty roster (a later-provisioned clinician would also receive those earlier users' pages)
    with hop("provision_clinician"):
        return provision_clinician(e2e_stack, email=new_email("clin"), phone="+15555550100", on_call=True)


async def _chat(stack, user, text):
    return await stack.edge("POST", "/api/v1/orchestrator/chat", token=user.access_token, json={"message": text})


def _assert_page_level(subject, max_level):
    """The page's level is one the episode reached: at most its max_level (a same-turn rise pages again, F8)."""
    match = _SUBJECT_RE.fullmatch(subject)
    assert match, "page subject is not the IF-09 form"
    assert LEVELS.index(match.group(1)) <= LEVELS.index(max_level), (match.group(1), max_level)


ACK_PATH = "/api/v1/notifications/crisis-alerts/ack/"


def _assert_page_without_phi(msg, user):
    """The page body is real (non-empty, carries the ack link), so the PHI-absence checks cannot pass vacuously."""
    assert msg.text and ACK_PATH in msg.text, "page body is empty or has no acknowledgement link"
    assert "alive anymore" not in msg.text and user.email not in msg.text


async def _acknowledge(stack, user_id, msg):
    """IF-09: GET shows a POST confirm form and changes nothing; POST acknowledges the episode."""
    path = urlsplit(extract_link(msg.text, ACK_PATH)).path
    page = await stack.edge("GET", path)
    assert page.status_code == 200
    assert _POST_FORM_RE.search(page.text), "GET does not show a POST confirm form"
    assert crisis_episodes(stack, user_id)[0]["status"] == "open", "GET changed the episode"
    assert (await stack.edge("POST", path)).status_code in (200, 303)
    assert crisis_episodes(stack, user_id)[0]["status"] == "acknowledged"


def _assert_resources(crisis):
    from solace_clinical import RESOURCES_VERSION, crisis_resources     # W1 (WS-12.2); imported inside the hop
    assert any("988" in res["contact"] for res in crisis["resources"])  # D-29 contact text is "Call or text 988" (F4)
    assert crisis["resources"] == crisis_resources("en-US") and crisis["resources_version"] == RESOURCES_VERSION


@pytest.mark.parametrize("kind", sorted(CRISIS_PHRASES))
async def test_rest_crisis_message_returns_resources_within_3s(e2e_stack, kind):
    user = await make_user(e2e_stack)
    with hop("rest_crisis_object"):
        t0 = time.monotonic()
        r = await _chat(e2e_stack, user, CRISIS_PHRASES[kind])
        elapsed = time.monotonic() - t0
        crisis = r.json()["crisis"]
        assert r.status_code == 200 and crisis["level"] in ("HIGH", "CRITICAL")
        _assert_resources(crisis)
        assert elapsed < 3.0


@pytest.mark.parametrize("kind", ["curly_apostrophe", "homicidal"])
async def test_ws_crisis_message_sends_safety_alert_before_reply(e2e_stack, kind):
    user = await make_user(e2e_stack)
    uri = e2e_stack.edge_url.replace("http", "ws") + f"/api/v1/orchestrator/ws/{uuid4()}"
    with hop("ws_connect"):
        # proxy=None: the edge is on 127.0.0.1, as Stack.edge's trust_env=False (websockets 15 reads proxy env vars)
        ws = await websockets.connect(uri, subprotocols=["solace.v1", f"bearer.{user.access_token}"], proxy=None)
        assert json.loads(await asyncio.wait_for(ws.recv(), 5.0))["type"] == "connected"   # IF-12 addition (WS-11.5, F5)
    try:
        with hop("ws_safety_alert"):
            t0 = time.monotonic()
            await ws.send(json.dumps({"type": "chat.message", "content": CRISIS_PHRASES[kind]}))
            first = json.loads(await asyncio.wait_for(ws.recv(), 3.0))         # the first frame after the send
            assert first["type"] == "safety.alert" and first["level"] in ("HIGH", "CRITICAL")
            assert time.monotonic() - t0 < 3.0
            _assert_resources(first)
            assert json.loads(await asyncio.wait_for(ws.recv(), 20.0))["type"] == "chat.response"
    finally:
        await ws.close()


async def test_crisis_pages_the_oncall_clinician_once_without_message_text(e2e_stack, oncall):
    user = await make_user(e2e_stack)
    with hop("rest_crisis_object"):
        assert (await _chat(e2e_stack, user, CRISIS_PHRASES["curly_apostrophe"])).json()["crisis"]
    with hop("clinician_page"):
        (episode,) = crisis_episodes(e2e_stack, user.id, timeout=60)          # one episode for this user's turn
        msg = wait_for_episode_page(e2e_stack, oncall.email, episode["alert_id"], timeout=60)
        _assert_page_level(msg.subject, crisis_episodes(e2e_stack, user.id)[0]["max_level"])   # W056-CRI-6
        _assert_page_without_phi(msg, user)
        await asyncio.sleep(15)
        pages = pages_for_episode(e2e_stack, oncall.email, episode["alert_id"])
        max_level = crisis_episodes(e2e_stack, user.id)[0]["max_level"]
        for page in pages:
            _assert_page_level(page.subject, max_level)
        assert len(pages) == len({p.subject.split("]")[0] for p in pages}) <= 2   # one page per level (a same-turn rise pages again, F8)


async def test_ack_link_acknowledges_the_episode(e2e_stack, oncall):
    user = await make_user(e2e_stack)
    with hop("rest_crisis_object"):
        assert (await _chat(e2e_stack, user, CRISIS_PHRASES["indirect"])).json()["crisis"]
    with hop("clinician_page"):
        (episode,) = crisis_episodes(e2e_stack, user.id, timeout=60)
        msg = wait_for_episode_page(e2e_stack, oncall.email, episode["alert_id"], timeout=60)
    with hop("ack_link"):
        await _acknowledge(e2e_stack, user.id, msg)       # W056-CRI-1
