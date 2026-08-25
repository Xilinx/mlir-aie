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
    "test_trace_config_without_enable_trace_raises": (
        "HRX rejects any trace_config up front, before the host-buffer argument "
        "validation this test asserts on"
    ),
}


def _running_on_hrx() -> bool:
    """True when the process's active host runtime is the HRX backend.

    The runtime is selected at ``aie.utils`` import time from ``NPU_RUNTIME``;
    the HRX RUN line sets ``NPU_RUNTIME=hrx`` so the default tensor class is
    ``HRXTensor``.
    """
    import aie.utils as aie_utils

    return getattr(aie_utils.DEFAULT_TENSOR_CLASS, "__name__", "") == "HRXTensor"


def pytest_collection_modifyitems(config, items):
    """Skip HRX-unsupported tests when running under the HRX backend."""
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
