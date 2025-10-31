# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s

import pytest
import aie.iron as iron
from pathlib import Path


def test_compile_ctx():
    with iron.compile_ctx(my_var=42):
        assert iron.get_compile_arg("my_var") == 42


def test_nested_compile_ctx():
    with iron.compile_ctx(my_var=42):
        with iron.compile_ctx(my_other_var=10):
            assert iron.get_compile_arg("my_var") == 42
            assert iron.get_compile_arg("my_other_var") == 10
        assert iron.get_compile_arg("my_other_var") is None


def test_compile_args_in_jit():
    @iron.jit(metaargs={"my_var": 42})
    def my_kernel():
        assert iron.get_compile_arg("my_var") == 42

    my_kernel()


def test_compile_args_hash():
    @iron.compileconfig(metaargs={"my_var": 42})
    def my_kernel_1():
        pass

    @iron.compileconfig(metaargs={"my_var": 43})
    def my_kernel_2():
        pass

    assert hash(my_kernel_1) != hash(my_kernel_2)


def test_mlir_file_generator():
    with open("test.mlir", "w") as f:
        f.write(
            """
module {
  func.func @main() {
    return
  }
}
"""
        )
    compilable = iron.compileconfig(mlir_generator=Path("test.mlir"))
    assert compilable.mlir_generator == Path("test.mlir")
    with pytest.raises(TypeError):
        compilable()


def test_function_generator():
    from .utils import _vector_vector_add_impl
    import numpy as np

    compilable = iron.compileconfig(mlir_generator=_vector_vector_add_impl)
    assert compilable.mlir_generator == _vector_vector_add_impl
    # This should not raise an error
    compilable(
        np.ones(16, dtype=np.int32),
        np.ones(16, dtype=np.int32),
        np.zeros(16, dtype=np.int32),
    )
