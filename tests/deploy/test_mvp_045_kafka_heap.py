# tests/deploy/test_mvp_045_kafka_heap.py
"""MVP-045: the broker heap plus JVM overhead fits its strict memory limit, and the healthcheck JVM is small."""
import re

MIB = 1024 * 1024

def _xmx_bytes(opts: str) -> int:
    value, unit = re.search(r"-Xmx(\d+)([mMgG])", opts).groups()
    return int(value) * (1024 * MIB if unit in "gG" else MIB)

def test_kafka_heap_fits_its_strict_limit(render):
    kafka = render("strict").services["kafka"]
    limit = int(kafka["deploy"]["resources"]["limits"]["memory"])
    assert _xmx_bytes(kafka["environment"]["KAFKA_HEAP_OPTS"]) + 384 * MIB <= limit

def test_kafka_healthcheck_runs_a_small_jvm(render):
    kafka = render("strict").services["kafka"]
    test = " ".join(kafka["healthcheck"]["test"])
    assert _xmx_bytes(test) <= 128 * MIB
    assert kafka["healthcheck"]["interval"] in ("30s", "1m0s")

# --- WS-03.4 additions ------------------------------------------------------------------------------
import pytest  # noqa: E402, I001 - kept below the brief's verbatim block

def test_zookeeper_heap_fits_its_strict_limit(render):
    zookeeper = render("strict").services["zookeeper"]
    limit = int(zookeeper["deploy"]["resources"]["limits"]["memory"])
    assert _xmx_bytes(zookeeper["environment"]["KAFKA_HEAP_OPTS"]) + 256 * MIB <= limit

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_kafka_healthcheck_never_inherits_the_broker_heap(render, profile):
    kafka = render(profile).services["kafka"]
    test = " ".join(kafka["healthcheck"]["test"])
    assert "KAFKA_HEAP_OPTS=" in test                      # the probe JVM sets its own heap inline
    assert (kafka["healthcheck"]["timeout"], kafka["healthcheck"]["start_period"]) == ("20s", "1m0s")

def test_kafka_ui_is_pinned(render):
    image = render("dev", profiles=["dev"]).services["kafka-ui"]["image"]
    assert ":" in image and not image.endswith(":latest"), image

@pytest.mark.integration
@pytest.mark.e2e
def test_strict_kafka_is_healthy_inside_its_memory_limit(compose_project, tmp_path):
    # a strict project reuses this test's unique project name; its teardown removes everything
    import json
    import shutil
    import subprocess

    from tests.deploy.conftest import PROFILES, ComposeProject, strict_env_file
    strict_env_file(tmp_path / "strict.env")
    strict = ComposeProject(compose_project.name, tmp_path / "strict.env", files=PROFILES["strict"])
    try:
        strict.up(["kafka"])  # --wait: zookeeper and the broker healthy under their strict limits
        cid = strict._call(["ps", "-q", "kafka"]).stdout.strip()
        # S603: a fixed docker argv (no shell) with the container id compose just reported
        inspect = subprocess.run([shutil.which("docker"), "inspect", cid], capture_output=True, text=True, check=True)  # noqa: S603
        state = json.loads(inspect.stdout)[0]
        assert state["HostConfig"]["Memory"] == 1024 * MIB
        assert state["State"]["Health"]["Status"] == "healthy" and not state["State"]["OOMKilled"]
        argv = strict.exec("kafka", "sh", "-c", "for p in /proc/[0-9]*/cmdline; do tr '\\0' ' ' < $p; echo; done")
        broker = [line for line in argv.splitlines() if "kafka.Kafka" in line]
        assert broker and "-Xmx512m" in broker[0] and "-Xmx1G" not in broker[0]
    finally:
        strict.down()
