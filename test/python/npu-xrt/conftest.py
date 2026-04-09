# conftest.py — shared pytest fixtures for npu-xrt tests
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
import pytest


@pytest.fixture(autouse=True)
def reset_iron_state():
    """Clear ExternalFunction._instances before and after every test.

    ``ExternalFunction._instances`` is a class-level global set that accumulates
    live instances for the ``@jit`` decorator to compile.  If a test fails during
    compilation, stale instances can remain and contaminate subsequent tests.
    Clearing it around each test ensures a clean slate without forcing expensive
    kernel recompilation (the in-process ``_kernel_cache`` is left intact so
    successive tests within the same session still benefit from caching).
    """
    from aie.iron.kernel import ExternalFunction

    ExternalFunction._instances.clear()
    yield
    ExternalFunction._instances.clear()
