# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""Scalar validation accepts typed SSA operands without accepting arrays."""

import numpy as np
import pytest

from aie import ir
from aie.dialects import memref
from aie.extras import types as T
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects.arith import constant
from aie.iron.kernel import ExternalFunction


def _kernel(dtype):
    return ExternalFunction(
        "scalar_arg", source_string="void scalar_arg() {}", arg_types=[dtype]
    )


@pytest.mark.parametrize(
    "dtype,type_ctor,value",
    [
        (np.int16, T.i16, 7),
        (np.int32, T.i32, 7),
        (np.float32, T.f32, 1.5),
        (np.int32, T.index, 7),
    ],
)
def test_matching_mlir_scalars(dtype, type_ctor, value):
    with mlir_mod_ctx():
        arg = constant(value, type_ctor())
        assert isinstance(arg, ir.Value)
        _kernel(dtype)._validate_arg(0, arg, dtype)


@pytest.mark.parametrize(
    "dtype,type_ctor,value",
    [
        (np.int32, T.i16, 7),
        (np.int32, T.f32, 1.5),
        (np.float32, T.i32, 7),
        (np.float32, T.index, 7),
    ],
)
def test_mismatched_mlir_scalars_are_rejected(dtype, type_ctor, value):
    with mlir_mod_ctx():
        arg = constant(value, type_ctor())
        with pytest.raises(ValueError, match="expected scalar"):
            _kernel(dtype)._validate_arg(0, arg, dtype)


@pytest.mark.parametrize("shape", [[], [2]])
def test_mlir_memref_is_not_a_scalar(shape):
    with mlir_mod_ctx():
        arg = memref.AllocOp(ir.MemRefType.get(shape, T.i32()), [], []).result
        with pytest.raises(ValueError, match="expected scalar"):
            _kernel(np.int32)._validate_arg(0, arg, np.int32)


@pytest.mark.parametrize("arg", [np.array(1), np.ones(2), [1], object()])
def test_host_arrays_and_objects_are_not_scalars(arg):
    with pytest.raises(ValueError, match="expected scalar"):
        _kernel(np.int32)._validate_arg(0, arg, np.int32)


@pytest.mark.parametrize("arg", [1, 1.5, np.int32(1), np.float32(1.5)])
def test_host_scalar_behavior_is_preserved(arg):
    _kernel(np.int32)._validate_arg(0, arg, np.int32)
