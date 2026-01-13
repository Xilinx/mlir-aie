# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc.

# RUN: %pytest %s

import pytest
from aie.utils.jit import jit, compileconfig
from aie.utils.compilable import Compilable
from aie.utils.npucallable import NPUCallable


def test_structure():
    # Mock function
    def my_kernel():
        pass

    # Test compileconfig
    c = compileconfig(my_kernel)
    assert isinstance(c, Compilable)

    # Test jit
    j = jit(my_kernel)
    assert isinstance(j, NPUCallable)

    # Test lazy NPUCallable
    assert j._xclbin_path is None
    assert j._compilable is not None
