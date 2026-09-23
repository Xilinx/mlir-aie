# conftest.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Shared fixtures for test/python/.

Resides at the directory root so pytest auto-discovers the fixtures for
every test module beneath it (not test/python/npu-xrt/, which has its
own conftest.py).
"""

import pytest
from aie.iron.device import NPU2Col1
from aie.iron.kernel import ExternalFunction
from aie.iron.kernels import _common as _kernels_common
from aie.utils.hostruntime import set_current_device


@pytest.fixture(autouse=True)
def _isolate_extern_state():
    """Reset process-wide ExternalFunction registry + _make_extern cache
    between tests so cross-test pollution doesn't trip auto-prefix-on-
    collision logic or _name assertions in name-variant tests."""
    ExternalFunction._instances.clear()
    _kernels_common._EXTERN_CACHE.clear()
    yield
    ExternalFunction._instances.clear()
    _kernels_common._EXTERN_CACHE.clear()


@pytest.fixture
def npu2_device():
    """Bind NPU2Col1 as the current device for the test, then restore what was bound.

    Safer than per-test try/finally: pytest unwinds the fixture even when
    the test body crashes mid-assertion. Modules whose every test needs it
    wrap this in an autouse fixture rather than restating it.
    """
    from aie.utils import get_current_device

    previous = get_current_device(probe_runtime=False)
    set_current_device(NPU2Col1())
    try:
        yield
    finally:
        set_current_device(previous)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "extensive: the full sweep (every case x edge data x seed); deselect with "
        '-m "not extensive"',
    )
    config.addinivalue_line(
        "markers",
        "supported_devices(*devices): the NPU generations a test's kernels exist "
        'for ("npu1", "npu2"); skipped elsewhere (see test/python/npu/conftest.py)',
    )
