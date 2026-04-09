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
    live instances for the ``@jit`` decorator to compile.  A failed compilation
    leaves stale entries that corrupt subsequent compilations.
    """
    from aie.iron.kernel import ExternalFunction

    ExternalFunction._instances.clear()
    yield
    ExternalFunction._instances.clear()
