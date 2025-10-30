# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s

import pytest
import aie.iron as iron

def test_metaprogramming_context():
    with iron.metaprogramming_ctx(my_var=42):
        assert iron.get_metaprogram("my_var") == 42

def test_nested_metaprogramming_context():
    with iron.metaprogramming_ctx(my_var=42):
        with iron.metaprogramming_ctx(my_other_var=10):
            assert iron.get_metaprogram("my_var") == 42
            assert iron.get_metaprogram("my_other_var") == 10
        assert iron.get_metaprogram("my_other_var") is None

def test_metaprogramming_in_jit():
    @iron.jit(metaprograms={"my_var": 42})
    def my_kernel():
        assert iron.get_metaprogram("my_var") == 42

    my_kernel()

def test_metaprogramming_hash():
    @iron.compileconfig(metaprograms={"my_var": 42})
    def my_kernel_1():
        pass

    @iron.compileconfig(metaprograms={"my_var": 43})
    def my_kernel_2():
        pass

    assert hash(my_kernel_1) != hash(my_kernel_2)
