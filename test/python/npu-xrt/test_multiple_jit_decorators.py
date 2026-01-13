# test_multiple_jit_decorators.py -*- Python -*-
#
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


def test_jit_on_jit():
    @jit(is_placed=False)
    @jit(is_placed=True)
    def my_kernel():
        pass

    # Outer jit should win (is_placed=False)
    assert isinstance(my_kernel, NPUCallable)
    assert my_kernel.compilable.is_placed == False
    # And it should wrap the original function, not another NPUCallable
    assert not isinstance(my_kernel.compilable.function, NPUCallable)
    assert my_kernel.compilable.function.__name__ == "my_kernel"


def test_jit_on_compileconfig():
    @jit(is_placed=False)
    @compileconfig(is_placed=True)
    def my_kernel():
        pass

    assert isinstance(my_kernel, NPUCallable)
    assert my_kernel.compilable.is_placed == False
    assert not isinstance(my_kernel.compilable.function, Compilable)


def test_compileconfig_on_jit():
    @compileconfig(is_placed=False)
    @jit(is_placed=True)
    def my_kernel():
        pass

    assert isinstance(my_kernel, Compilable)
    assert my_kernel.is_placed == False
    assert not isinstance(my_kernel.function, NPUCallable)


def test_compileconfig_on_compileconfig():
    @compileconfig(is_placed=False)
    @compileconfig(is_placed=True)
    def my_kernel():
        pass

    assert isinstance(my_kernel, Compilable)
    assert my_kernel.is_placed == False
    assert not isinstance(my_kernel.function, Compilable)
    assert my_kernel.function.__name__ == "my_kernel"


def test_alternating_decorators():
    @jit(is_placed=False)
    @compileconfig(is_placed=True)
    @jit(is_placed=True)
    def my_kernel():
        pass

    # Outer jit should win (is_placed=False)
    assert isinstance(my_kernel, NPUCallable)
    assert my_kernel.compilable.is_placed == False
    # And it should wrap the original function
    assert not isinstance(my_kernel.compilable.function, (NPUCallable, Compilable))
    assert my_kernel.compilable.function.__name__ == "my_kernel"
