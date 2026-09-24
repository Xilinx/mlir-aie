# conftest.py — shared pytest fixtures for the runtime-agnostic npu tests
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
from contextlib import contextmanager

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
        "benchmark: times a kernel and records benchmark-action rows; select "
        "with -m benchmark",
    )
    config._bench_rows = []
    config._bench_meta = {}


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
        "--bench-out",
        default=None,
        help="write benchmark-action rows here, if the session passes",
    )
    parser.addoption(
        "--bench-meta", default=None, help="write run provenance and any failures here"
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
        "--no-compile", action="store_true", help="skip the cold-rebuild measurement"
    )
    parser.addoption(
        "--baseline-sources",
        metavar="DIR",
        default=None,
        help="also measure every case with its kernels from DIR (a checkout root, "
        "as MLIR_AIE_KERNEL_SOURCES) and compare the raw output words; the pair "
        "goes to --bench-meta and the terminal summary, the rows stay this tree's",
    )


@pytest.fixture
def benchmark(request):
    """Record the benchmark-action rows a timed test produces.

    The row name is ``<case>/<metric>``, which is the series key
    ``benchmark-action`` charts on gh-pages; ``test_benchmark_series_names.py``
    pins the whole set, so a renamed case restarts a chart and has to say so.
    """
    config = request.config

    def record(case: str, metric: str, unit: str, value, span: str | None = None):
        row = {
            "name": f"{case}/{metric}",
            "unit": unit,
            "value": value,
            # Read now, not at fixture setup: preflight fills this in.
            "extra": config._bench_meta.get("provenance", ""),
        }
        if span:
            row["range"] = span
        config._bench_rows.append(row)

    return record


def pytest_sessionfinish(session, exitstatus):
    """Write the benchmark rows, but only from a session that passed.

    Timings from a run where some kernel returned the wrong answer are not
    worth charting, and a partial file would silently drop series. pytest's
    own exit status is the gate, so there is no second tally to keep in step
    with it. Meta is written either way -- when nothing was measured, that
    file is the only record of why.
    """
    import json
    from pathlib import Path

    config = session.config
    rows = getattr(config, "_bench_rows", [])
    meta = getattr(config, "_bench_meta", {})

    if meta_path := config.getoption("--bench-meta"):
        meta["exitstatus"] = int(exitstatus)
        meta["n_rows"] = len(rows)
        Path(meta_path).write_text(json.dumps(meta, indent=1))

    if out := config.getoption("--bench-out"):
        if exitstatus == 0 and rows:
            Path(out).write_text(json.dumps(rows, indent=1))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print the ``--baseline-sources`` comparison, one line per case."""
    baseline = getattr(config, "_bench_meta", {}).get("baseline")
    if not baseline:
        return
    tr = terminalreporter
    tr.section(f"baseline {baseline['sources']} -> this tree")
    tr.write_line(f"{'case':<48} {'cycles':>17} {'npu_us min':>19}  words")
    for name, c in baseline["cases"].items():
        cycles = "{} -> {}".format(*c["cycles"])
        npu = "{} -> {}".format(*c["npu_us_min"])
        words = "same" if not c["differing_words"] else f"{c['differing_words']} differ"
        tr.write_line(f"{name:<48} {cycles:>17} {npu:>19}  {words}")


def _device_generation() -> str | None:
    """``"npu1"`` / ``"npu2"`` for the device the tests will run on, or None."""
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
    return "npu2" if arch == "aie2p" else "npu1"


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
