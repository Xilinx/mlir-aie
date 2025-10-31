# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s

import pytest
from aie.iron.compileconfig import Compilable


def func1():
    pass


def func2():
    pass


def test_different_functions_hash_differently():
    compilable1 = Compilable(func1)
    compilable2 = Compilable(func2)
    assert hash(compilable1) != hash(compilable2)


def test_same_function_hashes_consistently():
    compilable1 = Compilable(func1)
    compilable2 = Compilable(func1)
    assert hash(compilable1) == hash(compilable2)


def test_compile_flags_affect_hash():
    compilable1 = Compilable(func1, compile_flags=["-O2"])
    compilable2 = Compilable(func1, compile_flags=["-O3"])
    assert hash(compilable1) != hash(compilable2)


def test_aiecc_flags_affect_hash():
    compilable1 = Compilable(func1, aiecc_flags=["--verbose"])
    compilable2 = Compilable(func1, aiecc_flags=[])
    assert hash(compilable1) != hash(compilable2)
