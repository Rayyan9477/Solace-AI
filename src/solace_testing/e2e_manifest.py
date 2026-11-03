"""Expected-failure manifest with hop tracking (MVP-039).

Acceptance suites are written red before the work that turns them green. A manifest
keeps such a test red *at a documented step* without hiding any other failure:

* ``hop(name)`` names the step a test is in. A failure inside it is re-raised as
  :class:`HopFailure` carrying ``.hop`` (innermost hop wins; the cause is chained).
* ``expected_failures.yaml`` (one per test directory, owned by that directory's owner)
  lists the tests expected to fail and the hop they must fail at::

      version: 1
      entries:
        - {test_id, expected_failing_hop, target_wave, reason, profiles?: [dev|strict], match?: exact|glob}
      shortcuts:
        - {name, remove_by_wave, reason}

  ``test_id`` is a pytest node id relative to the rootdir (the repo root), matched
  exactly by default, so ``[``, ``]``, ``*`` and ``?`` in a parametrized id are literal.
  With ``match: glob`` it is a pattern in which only ``*`` and ``?`` are wildcards
  (``[`` stays literal). A manifest only lists tests of its own directory. ``profiles``
  defaults to both profiles. Waves run from 0 to ``MAX_WAVE``.

An entry is *active* when ``SOLACE_MANIFEST_IGNORE`` is unset, it is not due
(``SOLACE_BARRIER_WAVE`` unset or lower than ``target_wave``) and its profiles contain
``SOLACE_E2E_PROFILE`` (default ``dev``). For an active entry, in both the setup and the
call phase:

* failing at the expected hop is reported XFAIL;
* failing at another hop, or outside any hop, FAILS:
  ``manifest expects hop '<x>', failed at '<y>'``;
* passing FAILS: ``XPASS(strict): <id> passed before wave <n>; remove its manifest entry``;
* a skip is left as a skip outside barrier runs (the manifest never turns anything into
  a pass).

A due entry is judged on its raw result: a failure FAILS with ``due manifest entry <id>
(target wave <n>) must pass at barrier wave <m>``. When ``SOLACE_BARRIER_WAVE`` is set,
a listed test (active or due) that is skipped FAILS, and a listed module skipped at
collection (e.g. ``importorskip``) fails the session: listed tests must run at a barrier.

In a directory that has a manifest, xfail is the manifest's alone: a static
``pytest.mark.xfail`` is a usage error, and an xfail marker added at run time or an
imperative ``pytest.xfail()`` never hides a failure. An entry whose test file was
collected but which matched no test fails the session (``stale manifest entry``). Due
entries are listed in the terminal summary. Schema and environment errors raise
:class:`ManifestError` (a ``pytest.UsageError``) naming the file and the key.

Environment:

* ``SOLACE_MANIFEST_IGNORE``: any non-empty value shows raw results.
* ``SOLACE_BARRIER_WAVE``: a non-negative integer. Any other value, including an empty
  one, is a usage error, so a misconfigured barrier can never fall back to applying the
  manifest.
* ``SOLACE_E2E_PROFILE``: ``dev`` (default) or ``strict``.

``tests/conftest.py`` imports ``pytest_configure``, ``pytest_collection_modifyitems`` and
``pytest_runtest_makereport`` so every test under ``tests/`` honours manifests; outside
the repo, load it with ``-p solace_testing.e2e_manifest``.
"""

from __future__ import annotations

import fnmatch
import os
import posixpath
from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import pytest
import yaml

__all__ = [
    "DEFAULT_MANIFEST_DIR",
    "MANIFEST_NAME",
    "MAX_WAVE",
    "PROFILES",
    "HopFailure",
    "ManifestEntry",
    "ManifestError",
    "Shortcut",
    "Verdict",
    "barrier_wave",
    "evaluate",
    "hop",
    "load_manifest",
    "pytest_collection_modifyitems",
    "pytest_configure",
    "pytest_runtest_makereport",
    "shortcut_allowed",
]

MANIFEST_NAME = "expected_failures.yaml"
PROFILES: tuple[str, ...] = ("dev", "strict")
DEFAULT_PROFILE = "dev"
MAX_WAVE = 5  # the programme runs waves W0..W5
MATCH_MODES: tuple[str, ...] = ("exact", "glob")
ENV_IGNORE = "SOLACE_MANIFEST_IGNORE"
ENV_BARRIER = "SOLACE_BARRIER_WAVE"
ENV_PROFILE = "SOLACE_E2E_PROFILE"

# src/solace_testing/e2e_manifest.py -> repo root -> tests/e2e
DEFAULT_MANIFEST_DIR = Path(__file__).resolve().parents[2] / "tests" / "e2e"

_SUMMARY_TITLE = "solace expected-failure manifest"
_REPORTER_NAME = "solace-e2e-manifest-reporter"
_JUDGED_ATTR = "_solace_manifest_judged"


# A manifest file or a manifest environment variable is invalid. pytest's UsageError is
# final, so this is an alias: pytest reports it as "ERROR: <file>: <key> ..." and exits 4.
ManifestError = pytest.UsageError


class HopFailure(AssertionError):  # noqa: N818 - public name fixed by the WS-05.3 interface
    """A failure inside ``hop(name)``; ``.hop`` names the step, ``__cause__`` is the error."""

    def __init__(self, hop: str, cause: BaseException | None = None) -> None:
        detail = "" if cause is None else f": {type(cause).__name__}: {cause}"
        super().__init__(f"hop '{hop}' failed{detail}")
        self.hop = hop


@contextmanager
def hop(name: str) -> Iterator[None]:
    """Name the step a test is in; a failure inside it becomes a :class:`HopFailure`.

    The innermost hop wins: a ``HopFailure`` from a nested hop passes through unchanged.
    pytest control flow (skip, xfail, exit) and non-``Exception`` signals such as
    ``KeyboardInterrupt`` or task cancellation pass through too; ``pytest.fail`` counts
    as a failure at the hop.
    """
    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError(f"hop name must be a non-empty string without surrounding spaces, got {name!r}")
    try:
        yield
    except HopFailure:
        raise
    except (pytest.skip.Exception, pytest.xfail.Exception, pytest.exit.Exception):
        raise
    except pytest.fail.Exception as exc:
        raise HopFailure(name, exc) from exc
    except Exception as exc:
        raise HopFailure(name, exc) from exc


@dataclass(frozen=True)
class ManifestEntry:
    """One expected failure: ``test_id`` fails at ``expected_failing_hop`` until ``target_wave``."""

    test_id: str
    expected_failing_hop: str
    target_wave: int
    reason: str
    profiles: tuple[str, ...] = PROFILES
    match: Literal["exact", "glob"] = "exact"
    source: Path | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "profiles", tuple(self.profiles))
        if self.match not in MATCH_MODES:
            raise ValueError(f"match must be one of {', '.join(MATCH_MODES)}, got {self.match!r}")

    @property
    def glob(self) -> bool:
        return self.match == "glob"

    @property
    def file_part(self) -> str:
        return self.test_id.split("::", 1)[0]

    def matches(self, nodeid: str) -> bool:
        return _id_match(nodeid, self.test_id, glob=self.glob)

    def matches_file(self, file_nodeid: str) -> bool:
        return _id_match(file_nodeid, self.file_part, glob=self.glob)


def _id_match(name: str, pattern: str, *, glob: bool) -> bool:
    """Exact match; with ``glob`` also an fnmatch where only ``*`` and ``?`` are wildcards."""
    return name == pattern or (glob and fnmatch.fnmatchcase(name, pattern.replace("[", "[[]")))


@dataclass(frozen=True)
class Shortcut:
    """A test-only shortcut allowed while listed and before ``remove_by_wave``."""

    name: str
    remove_by_wave: int
    reason: str
    source: Path | None = field(default=None, compare=False)


@dataclass(frozen=True)
class Verdict:
    """``xfail``: expected failure; ``fail``: fail the test; ``pass``: leave the report unchanged."""

    kind: Literal["xfail", "fail", "pass"]
    message: str


def evaluate(outcome: str, failed_hop: str | None, entry: ManifestEntry) -> Verdict:
    """Judge one raw phase outcome (``passed``/``failed``/``skipped``) of an active entry."""
    if outcome == "failed":
        if failed_hop is not None and failed_hop == entry.expected_failing_hop:
            return Verdict(
                "xfail",
                f"failed at the expected hop '{entry.expected_failing_hop}' "
                f"(target wave {entry.target_wave}): {entry.reason}",
            )
        shown = "<no hop>" if failed_hop is None else failed_hop
        return Verdict("fail", f"manifest expects hop '{entry.expected_failing_hop}', failed at '{shown}'")
    if outcome == "passed":
        return Verdict(
            "fail",
            f"XPASS(strict): {entry.test_id} passed before wave {entry.target_wave}; remove its manifest entry",
        )
    if outcome == "skipped":
        return Verdict("pass", "skipped; the manifest does not judge skips")
    raise ValueError(f"unknown outcome {outcome!r}; expected 'passed', 'failed' or 'skipped'")


# --------------------------------------------------------------------------- schema


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of keeping the last one."""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                None, None, f"unhashable key {key!r}", key_node.start_mark
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_keys(path: Path, where: str, data: Mapping[Any, Any], required: tuple[str, ...], optional: tuple[str, ...]) -> None:
    allowed = required + optional
    for key in data:
        if not isinstance(key, str) or key not in allowed:
            raise ManifestError(f"{path}: {where}: unknown key {key!r} (allowed: {', '.join(allowed)})")
    for key in required:
        if key not in data:
            raise ManifestError(f"{path}: {where}: missing required key '{key}'")


def _text(path: Path, where: str, data: Mapping[str, Any], key: str, *, exact: bool = True) -> str:
    value = data[key]
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{path}: {where}: key '{key}' must be a non-empty string, got {value!r}")
    if exact and value != value.strip():
        raise ManifestError(f"{path}: {where}: key '{key}' must not have surrounding spaces, got {value!r}")
    return value


def _wave(path: Path, where: str, data: Mapping[str, Any], key: str) -> int:
    value = data[key]
    if not _is_int(value) or value < 0 or value > MAX_WAVE:
        raise ManifestError(f"{path}: {where}: key '{key}' must be an integer from 0 to {MAX_WAVE}, got {value!r}")
    return value


def _list(path: Path, data: Mapping[str, Any], key: str) -> list[Any]:
    value = data.get(key, [])
    if not isinstance(value, list):
        raise ManifestError(f"{path}: key '{key}' must be a list, got {value!r}")
    return value


def _entry(path: Path, index: int, raw: object) -> ManifestEntry:
    where = f"entries[{index}]"
    if not isinstance(raw, dict):
        raise ManifestError(f"{path}: {where} must be a mapping, got {raw!r}")
    _check_keys(
        path, where, raw, ("test_id", "expected_failing_hop", "target_wave", "reason"), ("profiles", "match")
    )
    test_id = _text(path, where, raw, "test_id")
    if "::" not in test_id or not test_id.split("::", 1)[0] or not test_id.split("::", 1)[1]:
        raise ManifestError(
            f"{path}: {where}: key 'test_id' must be a pytest node id 'path/to/test_file.py::test_name', got {test_id!r}"
        )
    if "\\" in test_id or test_id.startswith(("/", "./", "../")):
        raise ManifestError(
            f"{path}: {where}: key 'test_id' must be relative to the repo root with forward slashes, got {test_id!r}"
        )
    profiles: tuple[str, ...] = PROFILES
    if "profiles" in raw:
        value = raw["profiles"]
        if (
            not isinstance(value, list)
            or not value
            or any(p not in PROFILES for p in value)
            or len(set(value)) != len(value)
        ):
            raise ManifestError(
                f"{path}: {where}: key 'profiles' must be a non-empty list of distinct values from "
                f"{', '.join(PROFILES)}, got {value!r}"
            )
        profiles = tuple(value)
    match = raw.get("match", "exact")
    if match not in MATCH_MODES:
        raise ManifestError(f"{path}: {where}: key 'match' must be one of {', '.join(MATCH_MODES)}, got {match!r}")
    return ManifestEntry(
        test_id=test_id,
        expected_failing_hop=_text(path, where, raw, "expected_failing_hop"),
        target_wave=_wave(path, where, raw, "target_wave"),
        reason=_text(path, where, raw, "reason", exact=False),
        profiles=profiles,
        match=match,
        source=path,
    )


def _shortcut(path: Path, index: int, raw: object) -> Shortcut:
    where = f"shortcuts[{index}]"
    if not isinstance(raw, dict):
        raise ManifestError(f"{path}: {where} must be a mapping, got {raw!r}")
    _check_keys(path, where, raw, ("name", "remove_by_wave", "reason"), ())
    return Shortcut(
        name=_text(path, where, raw, "name"),
        remove_by_wave=_wave(path, where, raw, "remove_by_wave"),
        reason=_text(path, where, raw, "reason", exact=False),
        source=path,
    )


def load_manifest(path: Path) -> tuple[list[ManifestEntry], list[Shortcut]]:
    """Load and schema-validate one manifest; raise :class:`ManifestError` naming file and key."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ManifestError(f"{path}: cannot read manifest ({exc})") from exc
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506 - SafeLoader subclass
    except yaml.YAMLError as exc:
        raise ManifestError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError(f"{path}: the manifest must be a mapping with keys 'version', 'entries', 'shortcuts'")
    _check_keys(path, "manifest", data, ("version",), ("entries", "shortcuts"))
    if not _is_int(data["version"]) or data["version"] != 1:
        raise ManifestError(f"{path}: key 'version' must be 1, got {data['version']!r}")
    entries = [_entry(path, i, raw) for i, raw in enumerate(_list(path, data, "entries"))]
    shortcuts = [_shortcut(path, i, raw) for i, raw in enumerate(_list(path, data, "shortcuts"))]
    for (i, a), (j, b) in combinations(enumerate(entries), 2):
        if a.test_id == b.test_id and set(a.profiles) & set(b.profiles):
            raise ManifestError(
                f"{path}: entries[{j}]: key 'test_id' {b.test_id!r} is already listed at entries[{i}] "
                f"for profile(s) {', '.join(sorted(set(a.profiles) & set(b.profiles)))}"
            )
    seen: dict[str, int] = {}
    for i, shortcut in enumerate(shortcuts):
        if shortcut.name in seen:
            raise ManifestError(
                f"{path}: shortcuts[{i}]: key 'name' {shortcut.name!r} is already listed at shortcuts[{seen[shortcut.name]}]"
            )
        seen[shortcut.name] = i
    return entries, shortcuts


# --------------------------------------------------------------------------- environment


def barrier_wave(environ: Mapping[str, str] | None = None) -> int | None:
    """``SOLACE_BARRIER_WAVE`` as an int, ``None`` when unset; any other value is an error."""
    env = os.environ if environ is None else environ
    if ENV_BARRIER not in env:
        return None
    raw = env[ENV_BARRIER]
    text = raw.strip()
    if not text or not text.isascii() or not text.isdigit():
        raise ManifestError(f"{ENV_BARRIER} must be a non-negative integer wave number, got {raw!r}")
    return int(text)


def _ignore_requested(environ: Mapping[str, str]) -> bool:
    return bool(environ.get(ENV_IGNORE, "").strip())


def _profile(environ: Mapping[str, str]) -> str:
    if ENV_PROFILE not in environ:
        return DEFAULT_PROFILE
    raw = environ[ENV_PROFILE]
    if raw.strip() not in PROFILES:
        raise ManifestError(f"{ENV_PROFILE} must be one of {', '.join(PROFILES)}, got {raw!r}")
    return raw.strip()


def shortcut_allowed(name: str, *, manifest_dir: Path | None = None) -> bool:
    """True only while ``name`` is listed and ``SOLACE_BARRIER_WAVE`` is unset or below its wave.

    ``manifest_dir`` defaults to ``tests/e2e``. A missing manifest allows nothing.
    """
    directory = DEFAULT_MANIFEST_DIR if manifest_dir is None else Path(manifest_dir)
    path = directory / MANIFEST_NAME
    if not path.is_file():
        return False
    _, shortcuts = load_manifest(path)
    barrier = barrier_wave()
    for shortcut in shortcuts:
        if shortcut.name == name:
            return barrier is None or barrier < shortcut.remove_by_wave
    return False


# --------------------------------------------------------------------------- pytest plugin


@dataclass
class _Manifest:
    path: Path
    entries: list[ManifestEntry]


@dataclass
class _RunState:
    ignore: bool
    barrier: int | None
    profile: str
    rootpath: Path
    collected_nodeids: set[str] = field(default_factory=set)
    skipped_nodeids: set[str] = field(default_factory=set)
    manifests: dict[Path, _Manifest | None] = field(default_factory=dict)
    indexed: bool = False
    applied: list[ManifestEntry] = field(default_factory=list)
    due: list[ManifestEntry] = field(default_factory=list)
    stale: list[ManifestEntry] = field(default_factory=list)
    barrier_skipped: list[tuple[ManifestEntry, str]] = field(default_factory=list)
    stale_check_skipped: str | None = None
    ignored_manifests: list[Path] = field(default_factory=list)

    @property
    def session_failures(self) -> bool:
        return bool(self.stale or self.barrier_skipped)

    def is_due(self, entry: ManifestEntry) -> bool:
        return self.barrier is not None and self.barrier >= entry.target_wave

    def shown(self, path: Path | None) -> str:
        if path is None:
            return "<unknown manifest>"
        try:
            return path.relative_to(self.rootpath).as_posix()
        except ValueError:
            return str(path)


_STATE_KEY = pytest.StashKey[_RunState]()
_ENTRY_KEY = pytest.StashKey[ManifestEntry]()  # active entry: judged against its hop
_DUE_KEY = pytest.StashKey[ManifestEntry]()  # due entry: judged on the raw result
_IN_MANIFEST_DIR_KEY = pytest.StashKey[bool]()  # the item's directory has a manifest


def _load_for_directory(state: _RunState, directory: Path) -> _Manifest | None:
    if directory in state.manifests:
        return state.manifests[directory]
    path = directory / MANIFEST_NAME
    manifest: _Manifest | None = None
    if path.is_file():
        entries, _ = load_manifest(path)
        try:
            rel = directory.relative_to(state.rootpath).as_posix()
        except ValueError:
            rel = None
        if rel is not None:
            rel = "" if rel == "." else rel
            for i, entry in enumerate(entries):
                if posixpath.dirname(entry.file_part) != rel:
                    raise ManifestError(
                        f"{path}: entries[{i}]: key 'test_id' {entry.test_id!r} must name a test in "
                        f"'{rel or '.'}' (node ids are relative to the rootdir {state.rootpath})"
                    )
        manifest = _Manifest(path=path, entries=entries)
    state.manifests[directory] = manifest
    return manifest


def _narrowed_files(config: pytest.Config) -> set[str]:
    """Rootdir-relative files whose collection was narrowed by a ``file::name`` argument."""
    narrowed: set[str] = set()
    for arg in config.args:
        if "::" not in arg:
            continue
        candidate = Path(arg.split("::", 1)[0])
        if not candidate.is_absolute():
            candidate = config.invocation_params.dir / candidate
        try:
            narrowed.add(Path(os.path.abspath(candidate)).relative_to(config.rootpath).as_posix())
        except ValueError:
            continue
    return narrowed


def pytest_configure(config: pytest.Config) -> None:
    if config.stash.get(_STATE_KEY, None) is not None:
        return
    state = _RunState(
        ignore=_ignore_requested(os.environ),
        barrier=barrier_wave(os.environ),
        profile=_profile(os.environ),
        rootpath=config.rootpath,
    )
    config.stash[_STATE_KEY] = state
    if not config.pluginmanager.has_plugin(_REPORTER_NAME):
        config.pluginmanager.register(_Reporter(state), _REPORTER_NAME)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]) -> None:
    """Attach active entries to items. Runs before ``-k``/``-m``/``--deselect`` narrow the list."""
    state = config.stash.get(_STATE_KEY, None)
    if state is None or state.indexed:
        return
    state.indexed = True

    files = {nodeid for nodeid in state.collected_nodeids if (state.rootpath / nodeid).is_file()}
    files.update(item.nodeid.split("::", 1)[0] for item in items)
    skipped_files = {nodeid for nodeid in state.skipped_nodeids if (state.rootpath / nodeid).is_file()}
    directories = {(state.rootpath / nodeid).parent for nodeid in files | skipped_files}
    directories.update(item.path.parent for item in items)
    manifest_dirs = {d for d in directories if (d / MANIFEST_NAME).is_file()}

    # xfail belongs to the manifest alone in its directory, in every mode (MVP-039).
    marked = []
    for item in items:
        if item.path.parent in manifest_dirs:
            item.stash[_IN_MANIFEST_DIR_KEY] = True
            if item.get_closest_marker("xfail") is not None:
                marked.append(item.nodeid)
    if marked:
        raise ManifestError(
            "use expected_failures.yaml, not pytest.mark.xfail, in a directory that has a manifest: "
            + ", ".join(sorted(marked))
        )

    if state.ignore:
        state.ignored_manifests = sorted(d / MANIFEST_NAME for d in manifest_dirs)
        return

    matched: set[int] = set()
    applied: dict[int, ManifestEntry] = {}
    due: dict[int, ManifestEntry] = {}
    for item in items:
        manifest = _load_for_directory(state, item.path.parent)
        if manifest is None:
            continue
        hits = [entry for entry in manifest.entries if entry.matches(item.nodeid)]
        for a, b in combinations(hits, 2):
            shared = set(a.profiles) & set(b.profiles)
            if shared:
                raise ManifestError(
                    f"{manifest.path}: {item.nodeid} matches both {a.test_id!r} and {b.test_id!r} "
                    f"for profile(s) {', '.join(sorted(shared))}"
                )
        matched.update(id(entry) for entry in hits)
        current = [entry for entry in hits if state.profile in entry.profiles]
        if not current:
            continue
        entry = current[0]
        if state.is_due(entry):
            item.stash[_DUE_KEY] = entry
            due[id(entry)] = entry
            continue
        item.stash[_ENTRY_KEY] = entry
        applied[id(entry)] = entry
    state.applied = list(applied.values())
    state.due = list(due.values())

    for directory in directories:
        _load_for_directory(state, directory)
    if state.barrier is not None:
        # A listed module skipped at collection (importorskip, allow_module_level skip) has
        # no items to judge; at a barrier it fails the session instead.
        for nodeid in sorted(skipped_files):
            manifest = state.manifests.get((state.rootpath / nodeid).parent)
            if manifest is None:
                continue
            for entry in manifest.entries:
                if state.profile in entry.profiles and entry.matches_file(nodeid):
                    state.barrier_skipped.append((entry, nodeid))
    if config.getoption("lf", False):
        state.stale_check_skipped = "--last-failed narrows collection"
        return
    narrowed = _narrowed_files(config)
    for manifest in state.manifests.values():
        if manifest is None:
            continue
        for entry in manifest.entries:
            if id(entry) in matched:
                continue
            collected = [f for f in files if entry.matches_file(f)]
            if collected and not any(f in narrowed for f in collected):
                state.stale.append(entry)


def _attach_failure(report: pytest.TestReport, message: str) -> None:
    longrepr = report.longrepr
    addsection = getattr(longrepr, "addsection", None)
    reprcrash = getattr(longrepr, "reprcrash", None)
    if callable(addsection) and reprcrash is not None:
        # Keep the original traceback; put the verdict in its own section and lead the
        # crash line with it, so the short summary and junit message show it first.
        addsection(_SUMMARY_TITLE, message)
        reprcrash.message = f"{message}; {reprcrash.message}"
        return
    if isinstance(longrepr, tuple) and len(longrepr) == 3:  # a skip report: (path, lineno, reason)
        original = f"skipped at {longrepr[0]}:{longrepr[1]}: {longrepr[2]}"
    else:
        original = "" if longrepr is None else str(longrepr)
    report.longrepr = message if not original else f"{message}\n\n{original}"


def _fail(report: pytest.TestReport, message: str) -> None:
    report.outcome = "failed"
    if hasattr(report, "wasxfail"):
        del report.wasxfail
    _attach_failure(report, message)


def _raw_outcome(call: pytest.CallInfo[None]) -> tuple[str, str | None]:
    """The phase result before any xfail handling: from the exception, never the report."""
    excinfo = call.excinfo
    if excinfo is None:
        return "passed", None
    if isinstance(excinfo.value, pytest.skip.Exception):
        return "skipped", None
    return "failed", excinfo.value.hop if isinstance(excinfo.value, HopFailure) else None


def _restore_raw(report: pytest.TestReport, outcome: str, failure_message: str | None) -> None:
    """Undo pytest's xfail handling: a failure fails (with ``failure_message``), a pass passes."""
    had_xfail = hasattr(report, "wasxfail")
    strict_xpass = (
        report.failed and isinstance(report.longrepr, str) and report.longrepr.startswith("[XPASS(strict)]")
    )
    if outcome == "failed":
        if failure_message is not None:
            _fail(report, failure_message)
        elif had_xfail:
            _fail(report, "use expected_failures.yaml, not pytest.mark.xfail or pytest.xfail(), in this directory")
    elif had_xfail or strict_xpass:
        # Only undo pytest's own xfail outcome; a pass failed by anything else stays failed.
        if had_xfail:
            del report.wasxfail
        report.outcome = "passed"
        report.longrepr = None


def _judge(item: pytest.Item, call: pytest.CallInfo[None], report: pytest.TestReport) -> None:
    if getattr(report, _JUDGED_ATTR, False):
        return
    if report.when == "teardown":
        # pytest's xfail also swallows teardown errors; in a manifest directory it must not.
        if item.stash.get(_IN_MANIFEST_DIR_KEY, False) and hasattr(report, "wasxfail") and call.excinfo is not None:
            _fail(report, "an xfail marker cannot hide a teardown failure in a directory that has a manifest")
            setattr(report, _JUDGED_ATTR, True)
        return
    if report.when not in ("setup", "call"):
        return
    if not item.stash.get(_IN_MANIFEST_DIR_KEY, False):
        return
    state = item.config.stash.get(_STATE_KEY, None)
    if state is None:
        return
    outcome, failed_hop = _raw_outcome(call)
    if outcome == "passed" and report.when != "call":
        return  # a passing setup is judged with its call phase
    entry = item.stash.get(_ENTRY_KEY, None)
    due = item.stash.get(_DUE_KEY, None)
    setattr(report, _JUDGED_ATTR, True)

    if outcome == "skipped":
        if (entry is not None or due is not None) and state.barrier is not None:
            _fail(report, f"manifest-listed test skipped during barrier wave {state.barrier}; listed tests must run")
        return
    if entry is not None:
        verdict = evaluate(outcome, failed_hop, entry)
        if verdict.kind == "xfail":
            report.outcome = "skipped"
            report.wasxfail = verdict.message
        elif verdict.kind == "fail":
            _fail(report, verdict.message)
        return
    if due is not None:
        _restore_raw(
            report,
            outcome,
            f"due manifest entry {due.test_id} (target wave {due.target_wave}) must pass at barrier wave {state.barrier}",
        )
        return
    # Unlisted (or not applied in this run) but in a manifest directory: no xfail of any kind.
    _restore_raw(report, outcome, None)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """Rewrite setup/call reports of active entries, after pytest's own xfail handling."""
    report = yield
    _judge(item, call, report)
    return report


class _Reporter:
    """Session-level hooks registered by ``pytest_configure``: collection, exit status, summary."""

    def __init__(self, state: _RunState) -> None:
        self.state = state

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if not report.nodeid or "::" in report.nodeid:
            return
        if report.passed:
            self.state.collected_nodeids.add(report.nodeid)
        elif report.skipped:
            self.state.skipped_nodeids.add(report.nodeid)

    @pytest.hookimpl(tryfirst=True)
    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        if self.state.session_failures and session.exitstatus in (
            pytest.ExitCode.OK,
            pytest.ExitCode.NO_TESTS_COLLECTED,
        ):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter: Any) -> None:
        state = self.state
        lines: list[tuple[str, dict[str, bool]]] = []
        if state.ignore:
            if state.ignored_manifests:
                lines.append((
                    f"{ENV_IGNORE} is set: {len(state.ignored_manifests)} manifest(s) not applied, raw results shown",
                    {"yellow": True},
                ))
        else:
            barrier = "unset" if state.barrier is None else str(state.barrier)
            if state.applied:
                lines.append((
                    f"{len(state.applied)} active entr{'y' if len(state.applied) == 1 else 'ies'} applied "
                    f"(profile {state.profile}, barrier wave {barrier})",
                    {},
                ))
            for entry in state.due:
                lines.append((
                    f"due manifest entry: {entry.test_id} (target wave {entry.target_wave}, barrier wave {barrier}, "
                    f"hop '{entry.expected_failing_hop}') judged on raw results; remove it from "
                    f"{state.shown(entry.source)}",
                    {"yellow": True},
                ))
            for entry in state.stale:
                lines.append((
                    f"stale manifest entry: {entry.test_id} in {state.shown(entry.source)}: its test file was "
                    f"collected but no test matched; fix or remove the entry",
                    {"red": True, "bold": True},
                ))
            for entry, nodeid in state.barrier_skipped:
                lines.append((
                    f"manifest-listed module skipped during barrier wave {barrier}: {nodeid} (entry {entry.test_id} "
                    f"in {state.shown(entry.source)}); listed tests must run",
                    {"red": True, "bold": True},
                ))
            if state.stale_check_skipped:
                lines.append((f"stale-entry check skipped: {state.stale_check_skipped}", {"yellow": True}))
        if not lines:
            return
        terminalreporter.write_sep("=", _SUMMARY_TITLE)
        for text, markup in lines:
            terminalreporter.write_line(text, **markup)
