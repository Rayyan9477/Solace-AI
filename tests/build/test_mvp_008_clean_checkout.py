"""MVP-008: a clean checkout of a tree builds and boots; tracked files never depend on untracked ones."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _closure(repo: Path, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "scripts/ci/check_tracked_closure.py"), "--repo", str(repo), *flags],
                          capture_output=True, text=True, timeout=120)


@pytest.fixture
def synthetic_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "r"
    (repo / "src" / "pkg").mkdir(parents=True)
    (repo / "src" / "pkg" / "__init__.py").write_text("")
    (repo / "src" / "pkg" / "a.py").write_text("from pkg import b\n")
    (repo / "src" / "pkg" / "b.py").write_text("X = 1\n")
    (repo / "Dockerfile").write_text("FROM scratch\nCOPY extra.txt /x\n")
    (repo / "extra.txt").write_text("x\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "src/pkg/__init__.py", "src/pkg/a.py", "Dockerfile")  # stage only; no commit
    return repo


def test_mvp_008_closure_reports_tracked_file_importing_untracked_module(synthetic_repo: Path) -> None:
    r = _closure(synthetic_repo, "--strict")
    assert r.returncode == 1
    assert "src/pkg/a.py -> src/pkg/b.py" in r.stdout and "Dockerfile -> extra.txt" in r.stdout


def test_mvp_008_closure_is_clean_once_dependencies_are_tracked(synthetic_repo: Path) -> None:
    _git(synthetic_repo, "add", "src/pkg/b.py", "extra.txt")
    r = _closure(synthetic_repo, "--strict")
    assert r.returncode == 0, r.stdout


def test_mvp_008_closure_report_mode_never_fails(synthetic_repo: Path) -> None:
    r = _closure(synthetic_repo, "--report")
    assert r.returncode == 0 and "src/pkg/a.py -> src/pkg/b.py" in r.stdout


def test_mvp_008_closure_reports_requirements_include(tmp_path: Path) -> None:
    repo = tmp_path / "q"
    repo.mkdir()
    (repo / "requirements.txt").write_text("-r base.txt\n")
    (repo / "base.txt").write_text("x==1\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "requirements.txt")
    r = _closure(repo, "--strict")
    assert r.returncode == 1 and "requirements.txt -> base.txt" in r.stdout


def test_mvp_008_clean_checkout_builds_every_registered_image_and_runs_probes() -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.ci import clean_checkout
    from scripts.ci.images import IMAGES

    plan = clean_checkout.plan("clean", IMAGES)
    assert [p.image.name for p in plan] == [i.name for i in IMAGES]
    migrate = next(p for p in plan if p.image.name == "migrate")
    assert migrate.build[:4] == ["docker", "build", "-f", "docker/migrate.Dockerfile"]
    assert migrate.build[-1] == "."
    assert [c[-1] for c in migrate.probes] == [
        "import importlib; importlib.import_module('scripts.db.migrate')",
        "import importlib; importlib.import_module('solace_infrastructure.database.schema_check')"]
    notification = next(p for p in plan if p.image.name == "notification-service")
    assert any("src.api" in c[-1] for c in notification.probes)
    for probe in migrate.probes:
        assert probe[:5] == ["docker", "run", "--rm", "--entrypoint", "python"]


def test_mvp_008_export_without_git_is_detected(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.ci import clean_checkout

    assert clean_checkout.has_git_context(ROOT) is True
    assert clean_checkout.has_git_context(tmp_path) is False


def _tracked(repo: Path, files: dict[str, str], untracked: dict[str, str] | None = None) -> Path:
    repo.mkdir(parents=True, exist_ok=True)
    for rel, text in {**files, **(untracked or {})}.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    _git(repo, "init", "-q")
    _git(repo, "add", *files)
    return repo


def test_mvp_008_r1_service_local_src_imports_resolve_inside_the_service(tmp_path: Path) -> None:
    repo = _tracked(tmp_path / "a", {
        "src/__init__.py": "",
        "services/notification-service/src/__init__.py": "",
        "services/notification-service/src/api.py": "from src.domain import NotificationService\n"},
        {"services/notification-service/src/domain.py": "class NotificationService: ...\n"})
    r = _closure(repo, "--strict")
    assert r.returncode == 1
    assert "services/notification-service/src/api.py -> services/notification-service/src/domain.py" in r.stdout


def test_mvp_008_r1_flat_service_imports_resolve_inside_the_service(tmp_path: Path) -> None:
    repo = _tracked(tmp_path / "b", {
        "services/analytics-service/src/main.py": "from aggregations import AnalyticsAggregator\n"},
        {"services/analytics-service/src/aggregations.py": "class AnalyticsAggregator: ...\n"})
    r = _closure(repo, "--strict")
    assert r.returncode == 1
    assert "services/analytics-service/src/main.py -> services/analytics-service/src/aggregations.py" in r.stdout


def test_mvp_008_r1_a_service_never_uses_another_services_roots(tmp_path: Path) -> None:
    repo = _tracked(tmp_path / "c", {
        "services/a/src/main.py": "from helper import X\n", "services/b/src/helper.py": "X = 1\n"})
    repo2 = _tracked(tmp_path / "d", {"services/a/src/main.py": "from helper import X\n"},
                     {"services/b/helper.py": "X = 1\n"})
    assert _closure(repo, "--strict").returncode == 0
    assert _closure(repo2, "--strict").returncode == 0  # old global roots included services/b and reported a pair


def test_mvp_008_r7_dockerignored_files_are_not_dependencies(tmp_path: Path) -> None:
    repo = _tracked(tmp_path / "e", {"Dockerfile": "FROM scratch\nCOPY src/ ./src/\n", ".dockerignore": "**/tests/\n",
                                     "src/a.py": "X = 1\n"}, {"src/tests/x.py": "X = 1\n", "src/new.py": "X = 2\n"})
    r = _closure(repo, "--strict")
    assert "src/tests/x.py" not in r.stdout
    assert "Dockerfile -> src/new.py" in r.stdout and r.returncode == 1


def test_mvp_008_r2_an_empty_or_unprobed_plan_is_refused() -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.ci import clean_checkout
    from scripts.ci.images import Image

    with pytest.raises(ValueError):
        clean_checkout.plan("t", [])
    with pytest.raises(ValueError, match="bare"):
        clean_checkout.plan("t", [Image("bare", "docker/x.Dockerfile", "", None)])


def test_mvp_008_r2_verdict_needs_every_expected_step_to_have_run() -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.ci import clean_checkout
    from scripts.ci.images import IMAGES

    steps = clean_checkout.plan("t", IMAGES)
    expected = len(steps) + sum(len(s.probes) for s in steps)
    assert clean_checkout.verdict([], [], compose=False, tests=False) == 1
    assert clean_checkout.verdict([("x", True)] * (expected - 1), steps, compose=False, tests=False) == 1
    assert clean_checkout.verdict([("x", True)] * expected, steps, compose=False, tests=False) == 0
    assert clean_checkout.verdict([("x", True)] * expected, steps, compose=True, tests=False) == 1
    bad = [("x", True)] * (expected - 1) + [("y", False)]
    assert clean_checkout.verdict(bad, steps, compose=False, tests=False) == 1


def test_mvp_008_r3_registry_comes_from_the_export(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.ci import clean_checkout

    (tmp_path / "scripts" / "ci").mkdir(parents=True)
    (tmp_path / "scripts" / "ci" / "images.py").write_text(
        "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass Image:\n    name: str\n    dockerfile: str\n"
        "    module: str\n    port: int | None\n    probes: tuple = ()\n    merged_src: bool = False\n"
        "IMAGES = (Image('only-in-export', 'docker/x.Dockerfile', 'm', 1),)\n"
        "def probe_env():\n    return {'K': 'v'}\n")
    images, env, source = clean_checkout.load_registry(tmp_path)
    assert [i.name for i in images] == ["only-in-export"] and env() == {"K": "v"} and "export" in source
    empty = tmp_path / "none"
    empty.mkdir()
    images, _, source = clean_checkout.load_registry(empty)
    assert "migrate" in [i.name for i in images] and "tool" in source


def test_mvp_008_r4_export_is_byte_identical_to_the_tree(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.ci import clean_checkout

    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "core.autocrlf", "true")
    (repo / "a.py").write_bytes(b"X = 1\nY = 2\n")
    _git(repo, "add", "a.py")
    tree = subprocess.run(["git", "-C", str(repo), "write-tree"], capture_output=True, text=True, check=True).stdout.strip()
    out = tmp_path / "out"
    clean_checkout.export_tree(tree, out, repo)
    assert (out / "a.py").read_bytes() == b"X = 1\nY = 2\n"


def test_mvp_008_r5_tests_step_env_is_isolated(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.ci import clean_checkout

    env = clean_checkout.tests_env(tmp_path, {"SOLACE_CLEAN_CHECKOUT_TREE": "x", "SOLACE_REQUIRE_REALINFRA": "1",
                                              "PATH": "p", "PYTHONPATH": "elsewhere"})
    assert not any(k.startswith("SOLACE_") for k in env)
    assert env["PYTHONPATH"] == os.pathsep.join([str(tmp_path), str(tmp_path / "src")])
    assert env["ENVIRONMENT"] == "test" and env["PATH"] == "p"
    assert "not realinfra" in clean_checkout.tests_command()


def test_mvp_008_r6_out_inside_the_repository_is_refused(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.ci import clean_checkout

    with pytest.raises(SystemExit):
        clean_checkout.resolve_out(str(ROOT / "scratch-export"), "abcdef0123")
    with pytest.raises(SystemExit):
        clean_checkout.resolve_out(str(ROOT), "abcdef0123")
    assert clean_checkout.resolve_out(str(tmp_path / "x"), "abcdef0123") == (tmp_path / "x").resolve()


def test_mvp_008_r8_tags_are_per_tree_and_builds_record_the_image_id() -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.ci import clean_checkout
    from scripts.ci.images import IMAGES

    import re

    tag = clean_checkout.default_tag("1dabe01159a0")
    assert re.fullmatch(r"clean-1dabe01-[0-9a-f]{8}", tag)
    assert re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", tag)
    assert clean_checkout.default_tag("1dabe01159a0") != tag  # unique per run
    cmd = clean_checkout.build_with_iid(clean_checkout.plan("t", IMAGES)[0].build, "id.txt")
    assert cmd[-1] == "." and cmd[-3:-1] == ["--iidfile", "id.txt"]


def test_mvp_008_n1_cleanup_untags_by_reference_only_when_the_ref_is_still_ours() -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.ci import clean_checkout

    calls: list[list[str]] = []
    ids = {"solace/a:t1": "sha256:aaa", "solace/b:t1": "sha256:other"}

    def fake(cmd: list[str], **_kw):
        calls.append(cmd)
        out = ids.get(cmd[-1], "") if cmd[:3] == ["docker", "image", "inspect"] else ""
        return subprocess.CompletedProcess(cmd, 0, stdout=out + "\n", stderr="")

    clean_checkout.cleanup([("solace/a:t1", "sha256:aaa"), ("solace/b:t1", "sha256:bbb")], run=fake)
    rmi = [c for c in calls if c[:2] == ["docker", "rmi"]]
    assert rmi == [["docker", "rmi", "solace/a:t1"]]  # b's ref now points at another image: left alone
    assert all("-f" not in c and not any(a.startswith("sha256:") for a in c) for c in rmi)


def test_mvp_008_main_drives_every_step_offline_and_cleans_up_by_reference(tmp_path: Path, monkeypatch, capsys) -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.ci import clean_checkout
    from scripts.ci.images import IMAGES

    resolved = "a" * 40
    ids: dict[str, str] = {}
    docker_calls: list[list[str]] = []
    exported: list[str] = []

    def fake_run(cmd, **_kw):
        cmd = [str(c) for c in cmd]
        out, rc = "", 0
        if cmd[0] == "git" and "rev-parse" in cmd:
            out = resolved + "\n"
        elif cmd[:2] == ["docker", "build"]:
            ref = cmd[cmd.index("-t") + 1]
            ids[ref] = "sha256:" + format(len(ids) + 1, "064x")
            Path(cmd[cmd.index("--iidfile") + 1]).write_text(ids[ref], encoding="utf-8")
        elif cmd[:3] == ["docker", "image", "inspect"]:
            out = ids.get(cmd[-1], "") + "\n"
        if cmd[0] == "docker":
            docker_calls.append(cmd)
        return subprocess.CompletedProcess(cmd, rc, stdout=out, stderr="")

    def fake_export(tree, out, repo=clean_checkout.REPO):
        exported.append(tree)
        out.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(clean_checkout.subprocess, "run", fake_run)
    monkeypatch.setattr(clean_checkout, "export_tree", fake_export)
    monkeypatch.setattr(clean_checkout, "load_registry", lambda _out: (list(IMAGES), clean_checkout.probe_env, "fake"))
    rc = clean_checkout.main(["--tree", "HEAD^{tree}", "--out", str(tmp_path / "x"), "--remove-images",
                              "--require-registry-match"])
    stdout = capsys.readouterr().out
    expected = len(IMAGES) + sum((1 if i.module else 0) + len(i.probes) for i in IMAGES)
    assert rc == 0, stdout[-2000:]
    assert stdout.strip().splitlines()[-1] == f"{expected}/{expected} steps ok"
    assert exported == [resolved]  # the tree-ish was resolved before use
    assert f"tag: clean-{resolved[:7]}-" in stdout
    rmi = [c for c in docker_calls if c[:2] == ["docker", "rmi"]]
    assert sorted(c[2] for c in rmi) == sorted(ids) and len(rmi) == len(IMAGES)
    assert all(len(c) == 3 and "-f" not in c and not c[2].startswith("sha256:") for c in rmi)
    runs = [c for c in docker_calls if c[:2] == ["docker", "run"]]
    assert runs and all(any(a.startswith("sha256:") for a in c) for c in runs)  # probes address the image id


def _one_file_tree(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "src-repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("x\n")
    _git(repo, "add", "a.txt")
    tree = subprocess.run(["git", "-C", str(repo), "write-tree"], capture_output=True, text=True, check=True).stdout.strip()
    return repo, tree


def test_mvp_008_cor1_an_export_with_a_private_git_dir_can_be_reused(tmp_path: Path) -> None:
    import stat

    sys.path.insert(0, str(ROOT))
    from scripts.ci import clean_checkout

    repo, tree = _one_file_tree(tmp_path)
    out = tmp_path / "out"
    clean_checkout.export_tree(tree, out, repo)
    clean_checkout._git_context(out)
    for p in (out / ".git").rglob("*"):  # Git for Windows leaves loose objects read-only
        if p.is_file():
            p.chmod(stat.S_IREAD)
    clean_checkout.export_tree(tree, out, repo)
    assert (out / "a.txt").exists() and (out / clean_checkout.MARKER).exists() and not (out / ".git").exists()


def test_mvp_008_cor1_a_failed_removal_keeps_the_marker(tmp_path: Path, monkeypatch) -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.ci import clean_checkout

    repo, tree = _one_file_tree(tmp_path)
    out = tmp_path / "out"
    clean_checkout.export_tree(tree, out, repo)
    (out / "stuck").mkdir()
    (out / "stuck" / "f.txt").write_text("x")

    # an open handle makes the file undeletable on Windows; elsewhere a refusing unlink stands in for it
    real_unlink = os.unlink

    def refuse(path, *a, **k):
        if str(path).endswith("f.txt"):
            raise PermissionError("undeletable")
        return real_unlink(path, *a, **k)

    monkeypatch.setattr(os, "unlink", refuse)
    monkeypatch.setattr(os, "remove", refuse)
    with pytest.raises(OSError):
        clean_checkout.export_tree(tree, out, repo)
    assert (out / clean_checkout.MARKER).exists()


def _offline_main(monkeypatch, export_images, flags: list[str], tmp_path: Path):
    from scripts.ci import clean_checkout

    calls: list[list[str]] = []

    def fake_run(cmd, **_kw):
        cmd = [str(c) for c in cmd]
        calls.append(cmd)
        out = "b" * 40 + "\n" if cmd[0] == "git" else ""
        if cmd[:2] == ["docker", "build"]:
            Path(cmd[cmd.index("--iidfile") + 1]).write_text("sha256:" + "1" * 64, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(clean_checkout.subprocess, "run", fake_run)
    monkeypatch.setattr(clean_checkout, "export_tree", lambda tree, out, repo=None: out.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(clean_checkout, "load_registry", lambda _o: (list(export_images), clean_checkout.probe_env, "fake"))
    return clean_checkout.main(["--tree", "HEAD", "--out", str(tmp_path / "x"), *flags]), calls


def test_mvp_008_tes1_registry_mismatch_fails_only_when_required(tmp_path: Path, monkeypatch, capsys) -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.ci.images import IMAGES

    rc, calls = _offline_main(monkeypatch, IMAGES[:3], ["--require-registry-match"], tmp_path)
    out = capsys.readouterr().out
    assert rc != 0 and "registries differ" in out and not [c for c in calls if c[:2] == ["docker", "build"]]
    rc, calls = _offline_main(monkeypatch, IMAGES[:3], [], tmp_path)
    out = capsys.readouterr().out
    assert rc == 0 and "warning: the export's image list differs" in out
    assert len([c for c in calls if c[:2] == ["docker", "build"]]) == 3


def test_mvp_008_tes2_resolve_tree_and_archive_get_an_env_without_git_variables(tmp_path: Path, monkeypatch) -> None:
    import io
    import tarfile

    sys.path.insert(0, str(ROOT))
    from scripts.ci import clean_checkout

    monkeypatch.setenv("GIT_DIR", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "foreign-index"))
    seen: list[tuple[str, dict]] = []
    buf = io.BytesIO()
    tarfile.open(fileobj=buf, mode="w").close()

    def fake_run(cmd, **kw):
        seen.append((" ".join(map(str, cmd)), kw.get("env")))
        return subprocess.CompletedProcess(cmd, 0, stdout=(buf.getvalue() if "archive" in cmd else "c" * 40 + "\n"), stderr=b"")

    monkeypatch.setattr(clean_checkout.subprocess, "run", fake_run)
    clean_checkout.resolve_tree("HEAD")
    clean_checkout.export_tree("c" * 40, tmp_path / "o", ROOT)
    assert len(seen) == 2
    for cmd, env in seen:
        assert env is not None and not any(k.startswith("GIT_") for k in env), cmd


def test_mvp_008_n2_a_tree_ish_is_resolved_to_one_tree_id(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT))
    import re

    from scripts.ci import clean_checkout

    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("x\n")
    _git(repo, "add", "a.txt")
    tree = subprocess.run(["git", "-C", str(repo), "write-tree"], capture_output=True, text=True, check=True).stdout.strip()
    assert clean_checkout.resolve_tree(tree[:10], repo) == tree
    assert re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", clean_checkout.default_tag(clean_checkout.resolve_tree(tree, repo)))
    with pytest.raises(SystemExit):
        clean_checkout.resolve_tree("no-such-tree", repo)


def test_mvp_008_c2_git_calls_ignore_the_operators_git_variables(tmp_path: Path, monkeypatch) -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.ci import clean_checkout

    other = tmp_path / "other"
    other.mkdir()
    _git(other, "init", "-q")
    foreign_index = tmp_path / "foreign-index"
    monkeypatch.setenv("GIT_INDEX_FILE", str(foreign_index))
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    out = tmp_path / "out"
    out.mkdir()
    (out / "f.txt").write_text("x\n")
    clean_checkout._git_context(out)
    assert (out / ".git" / "index").exists() and not foreign_index.exists()
    cfg = subprocess.run(["git", "config", "--file", str(other / ".git" / "config"), "core.bare"], capture_output=True, text=True)
    assert cfg.stdout.strip() == "false"
    assert not any(k.startswith(("SOLACE_", "GIT_")) for k in clean_checkout.tests_env(out, os.environ))


def test_mvp_008_c1_registry_match_flag_is_wired_everywhere() -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.ci import clean_checkout

    assert clean_checkout.build_parser().parse_args(["--tree", "t", "--require-registry-match"]).require_registry_match
    assert not clean_checkout.build_parser().parse_args(["--tree", "t"]).require_registry_match
    assert "--require-registry-match" in (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "require-registry-match" in (ROOT / "scripts/ci/check_clean_checkout.ps1").read_text(encoding="utf-8")


@pytest.mark.realinfra
def test_mvp_008_tree_builds_and_probes_from_clean_checkout(require_docker) -> None:
    tree = os.environ.get("SOLACE_CLEAN_CHECKOUT_TREE")
    if not tree:
        pytest.skip("set SOLACE_CLEAN_CHECKOUT_TREE to a checkpoint tree id")
    sys.path.insert(0, str(ROOT))
    from scripts.ci.images import IMAGES

    r = subprocess.run([sys.executable, str(ROOT / "scripts/ci/clean_checkout.py"), "--tree", tree, "--remove-images",
                        "--require-registry-match"], capture_output=True, text=True, timeout=3600)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    expected = len(IMAGES) + sum((1 if i.module else 0) + len(i.probes) for i in IMAGES)
    assert len(IMAGES) >= 10
    assert "warning: the export's image list differs" not in r.stdout
    assert r.stdout.strip().splitlines()[-1] == f"{expected}/{expected} steps ok", r.stdout[-1500:]
