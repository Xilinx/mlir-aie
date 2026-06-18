# conftest.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
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
    """Set the iron current device to NPU2Col1 for the test, then clear it.

    Safer than per-test try/finally: pytest unwinds the fixture even when
    the test body crashes mid-assertion.
    """
    set_current_device(NPU2Col1())
    yield
    set_current_device(None)
