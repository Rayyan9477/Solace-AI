"""MVP-039: the expected-failure manifest plugin with hop tracking.

A listed acceptance test may only be reported XFAIL when it fails at the exact hop its
manifest entry names. Failing elsewhere, failing outside any hop, or passing early all
fail the run, so the manifest can never hide a real crisis-path failure.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_mvp_039_evaluate_matrix():
    from solace_testing.e2e_manifest import ManifestEntry, evaluate
    e = ManifestEntry(test_id="x", expected_failing_hop="rest_crisis_object", target_wave=1, reason="r")
    assert evaluate("failed", "rest_crisis_object", e).kind == "xfail"
    other = evaluate("failed", "provision_clinician", e)
    assert other.kind == "fail" and "manifest expects hop 'rest_crisis_object'" in other.message
    assert evaluate("failed", None, e).kind == "fail"
    xpass = evaluate("passed", None, e)
    assert xpass.kind == "fail" and "XPASS(strict)" in xpass.message


def test_mvp_039_hop_names_the_failing_step():
    from solace_testing.e2e_manifest import HopFailure, hop
    with pytest.raises(HopFailure) as exc:
        with hop("outer"):
            with hop("register"):
                assert 1 == 2
    assert exc.value.hop == "register"


def test_mvp_039_hop_chains_the_cause_and_lets_pytest_control_flow_through():
    from solace_testing.e2e_manifest import HopFailure, hop
    with pytest.raises(HopFailure) as exc:
        with hop("login"):
            raise ValueError("401")
    assert isinstance(exc.value.__cause__, ValueError)
    with pytest.raises(HopFailure) as failed:
        with hop("ack_link"):
            pytest.fail("no ack link")
    assert failed.value.hop == "ack_link"
    with pytest.raises(pytest.skip.Exception):
        with hop("stack_up"):
            pytest.skip("docker unavailable")


SAMPLE = '''
from solace_testing.e2e_manifest import hop
def test_expected():
    with hop("a"): assert False
def test_other_hop():
    with hop("b"): assert False
def test_passes_early():
    assert True
def test_unlisted():
    assert True
'''
MANIFEST = '''version: 1
entries:
  - {test_id: "test_sample.py::test_expected", expected_failing_hop: a, target_wave: 1, reason: r}
  - {test_id: "test_sample.py::test_other_hop", expected_failing_hop: a, target_wave: 1, reason: r}
  - {test_id: "test_sample.py::test_passes_early", expected_failing_hop: a, target_wave: 1, reason: r}
shortcuts:
  - {name: verify_email_via_sql, remove_by_wave: 1, reason: r}
'''

_MANIFEST_ENV = ("SOLACE_BARRIER_WAVE", "SOLACE_MANIFEST_IGNORE", "SOLACE_E2E_PROFILE")


def _pytest(tmp_path, target, **env):
    base = {k: v for k, v in os.environ.items() if k not in _MANIFEST_ENV}
    return subprocess.run([sys.executable, "-m", "pytest", target, "-p", "solace_testing.e2e_manifest",  # noqa: S603
                           "-o", "addopts=", "-p", "no:cacheprovider", "-q"], cwd=tmp_path, capture_output=True, text=True, env={**base, **env})


def _write_sample(tmp_path):
    (tmp_path / "test_sample.py").write_text(SAMPLE)
    (tmp_path / "expected_failures.yaml").write_text(MANIFEST)


def _run(tmp_path, **env):
    _write_sample(tmp_path)
    return _pytest(tmp_path, "test_sample.py", **env)


def test_mvp_039_plugin_applies_strict_hop_xfail(tmp_path):
    out = _run(tmp_path).stdout
    assert "1 xfailed" in out and "2 failed" in out and "1 passed" in out
    assert "manifest expects hop 'a', failed at 'b'" in out and "XPASS(strict)" in out


def test_mvp_039_barrier_wave_makes_entries_due(tmp_path):
    out = _run(tmp_path, SOLACE_BARRIER_WAVE="1").stdout
    assert "2 failed" in out and "2 passed" in out and "xfailed" not in out
    assert "due manifest entry" in out


def test_mvp_039_ignore_shows_raw_results(tmp_path):
    out = _run(tmp_path, SOLACE_MANIFEST_IGNORE="1").stdout
    assert "2 failed" in out and "2 passed" in out


def test_mvp_039_stale_entry_fails_the_session(tmp_path):
    (tmp_path / "test_sample.py").write_text(SAMPLE)
    (tmp_path / "expected_failures.yaml").write_text(MANIFEST.replace("test_passes_early", "test_renamed"))
    # run as in _run without rewriting the manifest
    proc = _pytest(tmp_path, "test_sample.py")
    assert "stale manifest entry" in proc.stdout and proc.returncode != 0


def test_mvp_039_stale_entry_alone_fails_an_otherwise_green_session(tmp_path):
    (tmp_path / "test_green.py").write_text("def test_ok():\n    assert True\n")
    (tmp_path / "expected_failures.yaml").write_text(
        'version: 1\nentries:\n'
        '  - {test_id: "test_green.py::test_gone", expected_failing_hop: a, target_wave: 1, reason: r}\n'
        'shortcuts: []\n')
    proc = _pytest(tmp_path, "test_green.py")
    assert "1 passed" in proc.stdout and "stale manifest entry" in proc.stdout
    assert proc.returncode == 1


def test_mvp_039_node_id_selection_is_not_reported_stale(tmp_path):
    _write_sample(tmp_path)
    proc = _pytest(tmp_path, "test_sample.py::test_unlisted")
    assert "1 passed" in proc.stdout and "stale manifest entry" not in proc.stdout
    assert proc.returncode == 0


EDGE_SAMPLE = '''
import pytest
from solace_testing.e2e_manifest import hop

@pytest.fixture
def stack_up_fails():
    with hop("stack_up"):
        raise RuntimeError("compose up failed")

@pytest.fixture
def migrate_fails():
    with hop("migrate"):
        raise RuntimeError("migrate exited 1")

def test_setup_expected(stack_up_fails):
    pass

def test_setup_wrong_hop(migrate_fails):
    pass

def test_marker_wrong_hop(request):
    request.node.add_marker(pytest.mark.xfail(reason="a run-time xfail marker must not hide a wrong-hop failure"))
    with hop("b"): assert False

def test_strict_only():
    with hop("a"): assert False

@pytest.mark.parametrize("n", [1, 2])
def test_param(n):
    with hop("a"): assert False
'''
EDGE_MANIFEST = '''version: 1
entries:
  - {test_id: "test_edge.py::test_setup_expected", expected_failing_hop: stack_up, target_wave: 1, reason: r}
  - {test_id: "test_edge.py::test_setup_wrong_hop", expected_failing_hop: stack_up, target_wave: 1, reason: r}
  - {test_id: "test_edge.py::test_marker_wrong_hop", expected_failing_hop: a, target_wave: 1, reason: r}
  - {test_id: "test_edge.py::test_strict_only", expected_failing_hop: a, target_wave: 1, reason: r, profiles: [strict]}
  - {test_id: "test_edge.py::test_param*", match: glob, expected_failing_hop: a, target_wave: 2, reason: r}
shortcuts: []
'''


def test_mvp_039_setup_hops_markers_profiles_and_patterns(tmp_path):
    (tmp_path / "test_edge.py").write_text(EDGE_SAMPLE)
    (tmp_path / "expected_failures.yaml").write_text(EDGE_MANIFEST)
    dev = _pytest(tmp_path, "test_edge.py")
    assert "2 failed" in dev.stdout and "3 xfailed" in dev.stdout and "1 error" in dev.stdout, dev.stdout
    assert "manifest expects hop 'stack_up', failed at 'migrate'" in dev.stdout
    assert "manifest expects hop 'a', failed at 'b'" in dev.stdout
    assert dev.returncode == 1
    strict = _pytest(tmp_path, "test_edge.py", SOLACE_E2E_PROFILE="strict")
    assert "1 failed" in strict.stdout and "4 xfailed" in strict.stdout and "1 error" in strict.stdout, strict.stdout


def test_mvp_039_invalid_barrier_wave_is_a_usage_error(tmp_path):
    proc = _run(tmp_path, SOLACE_BARRIER_WAVE="W1")
    assert proc.returncode == 4 and "SOLACE_BARRIER_WAVE" in proc.stderr


def test_mvp_039_schema_errors_name_the_file_and_key(tmp_path):
    from solace_testing.e2e_manifest import load_manifest
    bad = tmp_path / "expected_failures.yaml"
    bad.write_text("version: 1\nentries:\n  - {test_id: 't.py::x', expected_failing_hop: a, target_wave: one, reason: r}\n")
    with pytest.raises(pytest.UsageError) as exc:
        load_manifest(bad)
    assert "expected_failures.yaml" in str(exc.value) and "target_wave" in str(exc.value)
    bad.write_text("version: 1\nentries: []\nshortcut: []\n")
    with pytest.raises(pytest.UsageError, match="shortcut"):
        load_manifest(bad)
    bad.write_text("version: 1\nentries:\n  - {test_id: 't.py::x', expected_failing_hop: a, target_wave: 6, reason: r}\n")
    with pytest.raises(pytest.UsageError) as wave:
        load_manifest(bad)
    assert "expected_failures.yaml" in str(wave.value) and "target_wave" in str(wave.value)
    bad.write_text("version: 1\nshortcuts:\n  - {name: s, remove_by_wave: 6, reason: r}\n")
    with pytest.raises(pytest.UsageError, match="remove_by_wave"):
        load_manifest(bad)
    bad.write_text("version: 1\nentries:\n  - {test_id: 't.py::x', match: regex, expected_failing_hop: a, target_wave: 1, reason: r}\n")
    with pytest.raises(pytest.UsageError) as mode:
        load_manifest(bad)
    assert "expected_failures.yaml" in str(mode.value) and "'match'" in str(mode.value)


def _write(tmp_path, test_file, source, entries):
    (tmp_path / test_file).write_text(source)
    body = "".join(f"\n  - {e}" for e in entries) if entries else " []"
    (tmp_path / "expected_failures.yaml").write_text(f"version: 1\nentries:{body}\n")


def test_mvp_039_due_entry_with_xfail_marker_fails_at_the_barrier(tmp_path):
    # A static marker is rejected at collection (below); one added at run time must still not hide the failure.
    _write(tmp_path, "test_due.py",
           "import pytest\nfrom solace_testing.e2e_manifest import hop\n"
           "def test_due(request):\n"
           "    request.node.add_marker(pytest.mark.xfail(reason='added at run time'))\n"
           "    with hop('b'): assert False\n",
           ['{test_id: "test_due.py::test_due", expected_failing_hop: a, target_wave: 1, reason: r}'])
    proc = _pytest(tmp_path, "test_due.py", SOLACE_BARRIER_WAVE="1")
    assert "1 failed" in proc.stdout and "xfailed" not in proc.stdout and proc.returncode == 1, proc.stdout
    assert "due manifest entry test_due.py::test_due (target wave 1) must pass at barrier wave 1" in proc.stdout


def test_mvp_039_run_time_xfail_cannot_hide_a_teardown_failure(tmp_path):
    _write(tmp_path, "test_td.py",
           "import pytest\n"
           "def _boom():\n    raise RuntimeError('finalizer failed')\n"
           "def test_due(request):\n"
           "    request.node.add_marker(pytest.mark.xfail(reason='added at run time'))\n"
           "    request.addfinalizer(_boom)\n",
           ['{test_id: "test_td.py::test_due", expected_failing_hop: a, target_wave: 1, reason: r}'])
    proc = _pytest(tmp_path, "test_td.py", SOLACE_BARRIER_WAVE="1")
    assert "1 error" in proc.stdout and "xfailed" not in proc.stdout and proc.returncode == 1, proc.stdout
    assert "an xfail marker cannot hide a teardown failure" in proc.stdout


def test_mvp_039_wildcard_characters_in_an_exact_id_are_literal(tmp_path):
    _write(tmp_path, "test_ws.py",
           "import pytest\nfrom solace_testing.e2e_manifest import hop\n"
           "@pytest.mark.parametrize('path', ['/ws?token=x', '/ws&token=x'])\n"
           "def test_ws(path):\n    with hop('a'): assert False\n",
           ['{test_id: "test_ws.py::test_ws[/ws?token=x]", expected_failing_hop: a, target_wave: 1, reason: r}'])
    proc = _pytest(tmp_path, "test_ws.py")
    assert "1 failed" in proc.stdout and "1 xfailed" in proc.stdout, proc.stdout
    assert "FAILED test_ws.py::test_ws[/ws&token=x]" in proc.stdout


def test_mvp_039_exact_entry_does_not_match_a_lookalike(tmp_path):
    _write(tmp_path, "test_l.py",
           "import pytest\nfrom solace_testing.e2e_manifest import hop\n"
           "@pytest.mark.parametrize('v', ['a*', 'abc'])\n"
           "def test_l(v):\n    with hop('a'): assert False\n",
           ['{test_id: "test_l.py::test_l[a*]", expected_failing_hop: a, target_wave: 1, reason: r}'])
    proc = _pytest(tmp_path, "test_l.py")
    assert "1 failed" in proc.stdout and "1 xfailed" in proc.stdout, proc.stdout
    assert "FAILED test_l.py::test_l[abc]" in proc.stdout


def test_mvp_039_xfail_markers_are_rejected_in_manifest_directories(tmp_path):
    _write(tmp_path, "test_static.py",
           "import pytest\n@pytest.mark.xfail(reason='hides failures')\ndef test_unlisted():\n    assert False\n", [])
    static = _pytest(tmp_path, "test_static.py")
    assert static.returncode == 4 and "use expected_failures.yaml, not pytest.mark.xfail" in static.stderr, static.stderr
    (tmp_path / "test_dynamic.py").write_text(
        "import pytest\ndef test_unlisted(request):\n"
        "    request.node.add_marker(pytest.mark.xfail(reason='added at run time'))\n    assert False\n")
    dynamic = _pytest(tmp_path, "test_dynamic.py")
    assert "1 failed" in dynamic.stdout and "xfailed" not in dynamic.stdout and dynamic.returncode == 1, dynamic.stdout


def test_mvp_039_listed_skips_fail_only_at_the_barrier(tmp_path):
    _write(tmp_path, "test_skipped.py",
           "import pytest\n"
           "@pytest.mark.skip(reason='docker unavailable')\ndef test_active():\n    assert False\n"
           "def test_due():\n    pytest.skip('docker unavailable')\n",
           ['{test_id: "test_skipped.py::test_active", expected_failing_hop: a, target_wave: 2, reason: r}',
            '{test_id: "test_skipped.py::test_due", expected_failing_hop: a, target_wave: 1, reason: r}'])
    outside = _pytest(tmp_path, "test_skipped.py")
    assert "2 skipped" in outside.stdout and outside.returncode == 0, outside.stdout
    barrier = _pytest(tmp_path, "test_skipped.py", SOLACE_BARRIER_WAVE="1")
    assert "skipped" not in barrier.stdout.splitlines()[-1] and barrier.returncode == 1, barrier.stdout
    assert barrier.stdout.count("manifest-listed test skipped during barrier wave 1; listed tests must run") >= 2


def test_mvp_039_listed_module_skipped_at_collection_fails_at_the_barrier(tmp_path):
    _write(tmp_path, "test_needs_dep.py",
           "import pytest\npytest.importorskip('ws053_module_that_does_not_exist')\n"
           "def test_listed():\n    assert False\n",
           ['{test_id: "test_needs_dep.py::test_listed", expected_failing_hop: a, target_wave: 1, reason: r}'])
    proc = _pytest(tmp_path, "test_needs_dep.py", SOLACE_BARRIER_WAVE="1")
    assert proc.returncode == 1 and "skipped during barrier wave 1" in proc.stdout, proc.stdout


def test_mvp_039_brackets_in_an_exact_id_are_literal(tmp_path):
    _write(tmp_path, "test_p.py",
           "import pytest\nfrom solace_testing.e2e_manifest import hop\n"
           "@pytest.mark.parametrize('ch', ['sms'])\ndef test_esc(ch):\n    with hop('a'): assert False\n"
           "def test_escs():\n    with hop('a'): assert False\n",
           ['{test_id: "test_p.py::test_esc[sms]", expected_failing_hop: a, target_wave: 1, reason: r}'])
    proc = _pytest(tmp_path, "test_p.py")
    assert "1 failed" in proc.stdout and "1 xfailed" in proc.stdout, proc.stdout
    assert "FAILED test_p.py::test_escs" in proc.stdout


def test_mvp_039_shortcut_expires_at_its_wave(tmp_path, monkeypatch):
    from solace_testing.e2e_manifest import shortcut_allowed
    (tmp_path / "expected_failures.yaml").write_text(MANIFEST)
    monkeypatch.delenv("SOLACE_BARRIER_WAVE", raising=False)
    assert shortcut_allowed("verify_email_via_sql", manifest_dir=tmp_path) is True
    monkeypatch.setenv("SOLACE_BARRIER_WAVE", "1")
    assert shortcut_allowed("verify_email_via_sql", manifest_dir=tmp_path) is False
    assert shortcut_allowed("unknown", manifest_dir=tmp_path) is False


def test_mvp_039_repo_manifests_are_valid():
    from solace_testing.e2e_manifest import load_manifest
    for path in (Path(__file__).resolve().parents[1]).rglob("expected_failures.yaml"):
        load_manifest(path)                                        # raises on schema errors
