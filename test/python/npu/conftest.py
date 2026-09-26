# conftest.py — shared pytest fixtures for the runtime-agnostic npu tests
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
from contextlib import contextmanager
import re
import xml.etree.ElementTree as ET

import pytest

# Tests in this directory run under both host runtimes (the %run_on_npu*_xrt%
# and %run_on_npu2_hrx% RUN lines). A few exercise features the HRX backend does
# not (yet) support; keep them running on XRT but skip them when the active
# runtime is HRX. Keyed by a substring of the pytest node id -> reason. These
# are HRX backend gaps worth a follow-up, not test bugs.
_HRX_UNSUPPORTED = {
    "test_variadic_tensor_count_checked_before_dispatch[65536]": (
        "HRX does not support trace capture; the untraced case still runs"
    ),
    "test_trace_config_without_enable_trace_raises": (
        "HRX rejects any trace_config up front, before the host-buffer argument "
        "validation this test asserts on"
    ),
}


def pytest_configure(config):
    """Register the markers these tests use.

    ``test/python/conftest.py`` registers them too, but the RUN lines invoke
    pytest on a file in *this* directory, which makes this directory the
    rootdir -- and pytest does not read a conftest.py above the rootdir.
    Without this, every run of these tests warns about an unknown mark.
    """
    config.addinivalue_line(
        "markers",
        "extensive: the full sweep (every case x edge data x seed); deselect with "
        '-m "not extensive"',
    )
    config.addinivalue_line(
        "markers",
        "supported_devices(*devices): the NPU generations a test's kernels exist "
        'for ("npu1", "npu2"); skipped elsewhere',
    )
    config.addinivalue_line(
        "markers",
        "perf: times a kernel and records benchmark-action rows; select with -m perf",
    )
    config._perf_rows = []
    config._perf_meta = {}
    config._error_report = {}


def _running_on_hrx() -> bool:
    """Return True when the process's active host runtime is the HRX backend.

    The runtime is selected at ``aie.utils`` import time from ``NPU_RUNTIME``;
    the HRX RUN line sets ``NPU_RUNTIME=hrx`` so the default tensor class is
    ``HRXTensor``.
    """
    import aie.utils as aie_utils

    return getattr(aie_utils.DEFAULT_TENSOR_CLASS, "__name__", "") == "HRXTensor"


def pytest_addoption(parser):
    parser.addoption(
        "--seeds",
        type=int,
        default=1,
        help="random seeds per case in the extensive kernel sweep",
    )
    parser.addoption(
        "--perf-out",
        default=None,
        help="write benchmark-action rows here, if the NPU checks pass",
    )
    parser.addoption(
        "--perf-meta", default=None, help="write run provenance and any failures here"
    )
    parser.addoption(
        "--correctness-results",
        default=None,
        help="only publish cases checked without failures in this extensive JUnit report",
    )
    parser.addoption("--warmup", type=int, default=10, help="untimed iterations")
    parser.addoption("--iters", type=int, default=50, help="timed iterations")
    parser.addoption(
        "--pmode",
        default="any",
        help="required device power mode; 'any' to accept whatever is set",
    )
    parser.addoption(
        "--no-cycles", action="store_true", help="skip the traced cycle-count run"
    )
    parser.addoption(
        "--baseline-sources",
        metavar="DIR",
        default=None,
        help="also measure every case with its kernels from DIR (a checkout root, "
        "as MLIR_AIE_KERNEL_SOURCES) and compare the raw output words, the "
        "check() cases untimed; the pair "
        "goes to --perf-meta and the terminal summary, the rows stay this tree's; "
        "a baseline that fails this tree's contract is reported, not failed",
    )
    parser.addoption(
        "--report-error",
        metavar="PATH",
        default=None,
        help="write each kernel test's error against its contract's reference "
        "run on float64 inputs (ulps, not correctly rounded, max abs/rel; each "
        "entry names the precision the reference reached), passing or not, to "
        "PATH as JSON, and summarize it after the run",
    )


@pytest.fixture
def record_perf(request):
    """Record the benchmark-action rows a timed test produces.

    The row name is ``<case>/<metric>``, which is the series key
    ``benchmark-action`` charts on gh-pages; ``test_perf_series_names.py``
    pins the whole set, so a renamed case restarts a chart and has to say so.
    """
    config = request.config

    def record(case: str, metric: str, unit: str, value, span: str | None = None):
        row = {
            "name": f"{case}/{metric}",
            "unit": unit,
            "value": value,
            # Read now, not at fixture setup: preflight fills this in.
            "extra": config._perf_meta.get("provenance", ""),
        }
        if span:
            row["range"] = span
        config._perf_rows.append(row)

    return record


@pytest.fixture
def report_error(request):
    """Record one run's ``cases.error_report`` entries, or None without the option.

    Keyed by ``<case>/<data>/s<seed>``; ``passed`` is the contract's verdict.
    """
    config = request.config
    if not config.getoption("--report-error"):
        return None

    def record(key: str, entries: list[dict], passed: bool):
        config._error_report[key] = {"passed": passed, "outputs": entries}

    return record


def _checked_cases(path):
    """Read the extensive sweep, excluding a case if any input or seed failed."""
    tests = list(ET.parse(path).iter("testcase"))
    if not tests:
        raise ValueError("correctness report contains no tests")
    checked, rejected, failed = set(), set(), []
    for test in tests:
        name = test.get("name", "")
        match = re.fullmatch(r"test_kernel_extensive\[(.+)/[^/]+/s\d+\]", name)
        bad = test.find("failure") is not None or test.find("error") is not None
        if bad:
            if not match:
                raise ValueError(f"unmapped correctness failure: {name}")
            rejected.add(match[1])
            failed.append(f"{test.get('classname', '')}::{name}")
        elif match and test.find("skipped") is None:
            checked.add(match[1])
    return checked - rejected, failed


def pytest_sessionfinish(session, exitstatus):
    """Write the performance rows once the NPU checks have passed.

    A kernel that returns the wrong answer records nothing -- its test raises
    before timing. When given the extensive correctness report, also exclude
    cases that failed an edge input or were not checked there. A failed
    kernel's series shows a gap for this run. What a partial file
    cannot survive is a bad device: if preflight (power mode) or the
    measurement sanity check failed, no number from the run is trustworthy
    and nothing is written; a sanity check that was selected must pass. A
    ``-k`` that deselects it leaves ``measurement_sane`` null in the meta and
    still writes the rows. Meta is written either way and lists the failed
    tests, so a missing series or an empty run is explained. The
    ``--report-error`` JSON is written whatever happened.
    """
    import json
    from pathlib import Path

    config = session.config
    rows = getattr(config, "_perf_rows", [])
    meta = getattr(config, "_perf_meta", {})
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    stats = reporter.stats if reporter else {}
    failed = sorted({r.nodeid for k in ("failed", "error") for r in stats.get(k, [])})
    if correctness := config.getoption("--correctness-results"):
        try:
            checked, correctness_failed = _checked_cases(correctness)
            rows = [r for r in rows if r["name"].rsplit("/", 1)[0] in checked]
            failed = sorted(set(failed) | set(correctness_failed))
        except (OSError, ET.ParseError, ValueError) as exc:
            meta["correctness_error"] = str(exc)
            rows = []

    sanity_selected = any(i.name == "test_measurement_is_sane" for i in session.items)
    meta.setdefault("measurement_sane", None)
    if meta_path := config.getoption("--perf-meta"):
        meta["exitstatus"] = int(exitstatus)
        meta["n_rows"] = len(rows)
        meta["failed"] = failed
        Path(meta_path).write_text(json.dumps(meta, indent=1))

    sane = meta["measurement_sane"]
    npu_ok = "preflight" in meta and (
        sane is True or (sane is None and not sanity_selected)
    )
    completed = exitstatus in (pytest.ExitCode.OK, pytest.ExitCode.TESTS_FAILED)
    if out := config.getoption("--perf-out"):
        if npu_ok and completed and rows:
            Path(out).write_text(json.dumps(rows, indent=1))
    if out := config.getoption("--report-error"):
        Path(out).write_text(json.dumps(config._error_report, indent=1))


def _accuracy_line(entry: dict) -> str:
    return (
        f"{entry['dtype']} vs {entry['reference']}: "
        f"{entry['not_correctly_rounded']}/{entry['n']} not correctly rounded, "
        f"max {entry['max_ulp']} ulp ({entry['max_ulp_error']:.3g} exact), "
        f"mean {entry['mean_ulp']:.3g}"
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print the ``--report-error`` stats and the ``--baseline-sources`` comparison.

    Each baseline arm shows its min..max over n calls (cycles) or iterations
    (npu_us), so "the candidate's max is below the base's min" reads off one
    line, and its error against the reference (``cases.error_report``)
    below the words.
    """
    tr = terminalreporter
    if errors := getattr(config, "_error_report", None):
        tr.section(f"error vs reference -> {config.getoption('--report-error')}")
        for key, run in errors.items():
            verdict = "" if run["passed"] else "  [FAILED]"
            if not run["outputs"]:
                tr.write_line(f"{key}  no floating output{verdict}")
            for entry in run["outputs"]:
                tr.write_line(
                    f"{key}[{entry['output']}]  {_accuracy_line(entry)}{verdict}"
                )
    baseline = getattr(config, "_perf_meta", {}).get("baseline")
    if not baseline:
        return

    def span(r):
        return f"{r['min']}..{r['max']} n={r['n']}" if r else "-"

    tr.section(f"baseline {baseline['sources']} -> this tree")
    tr.write_line(f"this tree's kernels: {baseline['current_sources']}")
    if baseline["warning"]:
        tr.write_line(f"warning: {baseline['warning']}", yellow=True, bold=True)
    for name, c in baseline["cases"].items():
        words = "same" if not c["differing_words"] else f"{c['differing_words']} differ"
        tr.write_line(f"{name}  words {words}")
        if c.get("current_failed"):
            tr.write_line(
                f"  this tree fails its contract: {c['current_failed']}", red=True
            )
        if c["baseline_failed"]:
            tr.write_line(
                f"  baseline fails this tree's contract: {c['baseline_failed']}",
                yellow=True,
            )
        if stack := c.get("baseline_stack"):
            tr.write_line(
                f"  baseline built with a {stack[1]} B stack: it needs more than "
                f"the {stack[0]} B this tree's contract gives",
                yellow=True,
            )
        base, cur = c.get("accuracy", ([], []))
        for b, a in zip(base, cur):
            tr.write_line(f"  error[{b['output']}] {_accuracy_line(b)}")
            tr.write_line(f"  {'':>8} -> {_accuracy_line(a)}")
        for metric in ("cycles", "npu_us"):
            if not any(c[f"{metric}_range"]):
                continue
            base, cur = c[f"{metric}_range"]
            tr.write_line(f"  {metric:<7} {span(base):>26} -> {span(cur)}")


def _device_generation() -> str | None:
    """``"npu1"`` / ``"npu2"`` for the device the tests will run on, or None."""
    from aie.iron.kernels._common import ARCH_TRAITS
    from aie.utils import get_current_device
    from aie.utils.compile.utils import resolve_target_arch

    # ``resolve_target_arch(None)`` deliberately defaults to "aie2" for callers
    # that don't care about device-specific codegen; here it would misclassify
    # "no device" (e.g. a static-checks runner with no NPU attached) as npu1
    # and skip every npu2-only case. Bail out before that default kicks in.
    device = get_current_device()
    if device is None:
        return None
    try:
        arch = resolve_target_arch(device)
    except Exception:  # noqa: BLE001 - unrecognized device: nothing to skip on
        return None
    return ARCH_TRAITS[arch].device


def pytest_collection_modifyitems(config, items):
    """Skip HRX-unsupported tests under HRX, and device-restricted tests elsewhere.

    ``@pytest.mark.supported_devices("npu2")`` names the generations a
    test's kernels exist for (IRON's marker of the same name); the test is
    skipped on any other device.
    """
    generation = _device_generation()
    for item in items:
        marker = item.get_closest_marker("supported_devices")
        if marker and generation and generation not in marker.args:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"kernel exists for {marker.args}, not {generation}"
                )
            )
    if not _running_on_hrx():
        return
    for item in items:
        for pattern, reason in _HRX_UNSUPPORTED.items():
            if pattern in item.nodeid:
                item.add_marker(pytest.mark.skip(reason=f"HRX unsupported: {reason}"))
                break


@pytest.fixture
def skip_on_f32_failure():
    """Fixture that returns a context manager for skipping f32 Peano failures.

    The Peano backend has a known stack-overflow bug when compiling certain
    f32 kernels.  Rather than marking those tests as ``xfail`` (which hides
    the issue permanently), request this fixture and wrap the test body so
    the test is skipped when the failure actually occurs and automatically
    starts passing if Peano fixes the bug.

    Usage::

        def test_something(dtype, skip_on_f32_failure):
            with skip_on_f32_failure():
                run_my_kernel(dtype=dtype)
    """

    @contextmanager
    def _guard():
        try:
            yield
        except Exception as exc:
            pytest.skip(f"Skipping: f32 Peano compilation/execution failure: {exc}")

    return _guard


@pytest.fixture(autouse=True)
def reset_iron_state():
    """Clear ExternalFunction._instances before and after every test.

    ``ExternalFunction._instances`` is a class-level global set that accumulates
    live instances for the ``@jit`` decorator to compile.  A failed compilation
    leaves stale entries that corrupt subsequent compilations.
    """
    from aie.iron.kernel import ExternalFunction

    ExternalFunction._instances.clear()
    yield
    ExternalFunction._instances.clear()


@pytest.fixture(autouse=True)
def bind_current_device():
    """Bind the device before each test so arch guards can read it.

    ``_detect_arch()`` reads the current device without probing, which is right
    where kernel factories call it -- inside a generator, by which point
    compilation has bound one. A test body runs before that, so an arch guard
    at the top of one saw no device and read ``aie2``, silently skipping every
    aie2p test on an aie2p board. Probing once here gives the guards the same
    answer the generator would get.
    """
    from aie.utils import get_current_device, set_current_device

    if get_current_device(probe_runtime=False) is None:
        try:
            set_current_device(get_current_device(probe_runtime=True))
        except (RuntimeError, ValueError, AttributeError):
            pass  # No device to probe; arch guards fall back as before.
    yield
