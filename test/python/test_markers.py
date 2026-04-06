# test_markers.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Unit tests for Compile[T], In, Out, InOut annotation markers — no NPU required."""

import inspect
from typing import get_args, get_origin

import pytest

from aie.iron.compile.markers import Compile, In, InOut, Out
from aie.iron.compile.compilabledesign import (
    _is_compile_param,
    _is_tensor_param,
    _split_params,
)

# ---------------------------------------------------------------------------
# Compile[T] — generic parameterisation
# ---------------------------------------------------------------------------


def test_compile_int_origin_is_compile():
    assert get_origin(Compile[int]) is Compile


def test_compile_str_origin_is_compile():
    assert get_origin(Compile[str]) is Compile


def test_compile_float_origin_is_compile():
    assert get_origin(Compile[float]) is Compile


def test_compile_type_arg_preserved():
    assert get_args(Compile[int]) == (int,)
    assert get_args(Compile[str]) == (str,)


def test_bare_compile_is_not_parameterised():
    # Bare Compile has no origin.
    assert get_origin(Compile) is None


def test_compile_different_type_args_are_distinct():
    # Compile[int] and Compile[str] are different objects (even though their
    # semantics only differ by type-checker; at runtime they share the same origin).
    assert Compile[int] is not Compile[str]


# ---------------------------------------------------------------------------
# _is_compile_param
# ---------------------------------------------------------------------------


def test_is_compile_param_with_int():
    assert _is_compile_param(Compile[int]) is True


def test_is_compile_param_with_str():
    assert _is_compile_param(Compile[str]) is True


def test_is_compile_param_bare():
    assert _is_compile_param(Compile) is True


def test_is_compile_param_rejects_in():
    assert _is_compile_param(In) is False


def test_is_compile_param_rejects_out():
    assert _is_compile_param(Out) is False


def test_is_compile_param_rejects_inout():
    assert _is_compile_param(InOut) is False


def test_is_compile_param_rejects_builtin_types():
    assert _is_compile_param(int) is False
    assert _is_compile_param(float) is False
    assert _is_compile_param(str) is False


def test_is_compile_param_rejects_none():
    assert _is_compile_param(None) is False


def test_is_compile_param_rejects_empty():
    assert _is_compile_param(inspect.Parameter.empty) is False


# ---------------------------------------------------------------------------
# _is_tensor_param
# ---------------------------------------------------------------------------


def test_is_tensor_param_in():
    assert _is_tensor_param(In) is True


def test_is_tensor_param_out():
    assert _is_tensor_param(Out) is True


def test_is_tensor_param_inout():
    assert _is_tensor_param(InOut) is True


def test_is_tensor_param_rejects_compile():
    assert _is_tensor_param(Compile[int]) is False
    assert _is_tensor_param(Compile) is False


def test_is_tensor_param_rejects_scalars():
    assert _is_tensor_param(int) is False
    assert _is_tensor_param(float) is False
    assert _is_tensor_param(str) is False


def test_is_tensor_param_rejects_none():
    assert _is_tensor_param(None) is False


def test_is_tensor_param_rejects_empty():
    assert _is_tensor_param(inspect.Parameter.empty) is False


# ---------------------------------------------------------------------------
# In / Out / InOut — distinct classes
# ---------------------------------------------------------------------------


def test_tensor_markers_are_distinct():
    assert In is not Out
    assert In is not InOut
    assert Out is not InOut


def test_tensor_markers_are_not_compile():
    assert In is not Compile
    assert Out is not Compile
    assert InOut is not Compile


def test_tensor_markers_are_classes():
    assert isinstance(In, type)
    assert isinstance(Out, type)
    assert isinstance(InOut, type)


# ---------------------------------------------------------------------------
# _split_params — comprehensive signature introspection
# ---------------------------------------------------------------------------


def test_split_params_all_compile():
    def f(*, M: Compile[int], K: Compile[int]):
        pass

    compile_params, tensor_params, scalar_params = _split_params(f)
    assert compile_params == ["M", "K"]
    assert tensor_params == []
    assert scalar_params == []


def test_split_params_all_tensor():
    def f(a: In, b: Out, c: InOut):
        pass

    compile_params, tensor_params, scalar_params = _split_params(f)
    assert compile_params == []
    assert tensor_params == ["a", "b", "c"]
    assert scalar_params == []


def test_split_params_all_scalar_annotated():
    def f(x: int, y: float, z: str):
        pass

    compile_params, tensor_params, scalar_params = _split_params(f)
    assert compile_params == []
    assert tensor_params == []
    assert scalar_params == ["x", "y", "z"]


def test_split_params_all_unannotated():
    def f(x, y, z):
        pass

    compile_params, tensor_params, scalar_params = _split_params(f)
    assert compile_params == []
    assert tensor_params == []
    assert scalar_params == ["x", "y", "z"]


def test_split_params_no_params():
    def f():
        pass

    compile_params, tensor_params, scalar_params = _split_params(f)
    assert compile_params == []
    assert tensor_params == []
    assert scalar_params == []


def test_split_params_mixed_all_three():
    def f(a: In, b: Out, alpha: float, *, M: Compile[int], N: Compile[int]):
        pass

    compile_params, tensor_params, scalar_params = _split_params(f)
    assert compile_params == ["M", "N"]
    assert tensor_params == ["a", "b"]
    assert scalar_params == ["alpha"]


def test_split_params_inout_goes_in_tensor():
    def f(x: InOut, *, M: Compile[int]):
        pass

    compile_params, tensor_params, scalar_params = _split_params(f)
    assert tensor_params == ["x"]
    assert compile_params == ["M"]


def test_split_params_preserves_declaration_order_for_tensors():
    """Tensor params must come out in the same order as the function signature."""

    def f(c: Out, a: In, b: InOut):
        pass

    _, tensor_params, _ = _split_params(f)
    assert tensor_params == ["c", "a", "b"]


def test_split_params_preserves_declaration_order_for_compile():
    def f(*, N: Compile[int], M: Compile[int], K: Compile[int]):
        pass

    compile_params, _, _ = _split_params(f)
    assert compile_params == ["N", "M", "K"]


def test_split_params_compile_with_default():
    """Parameters with defaults are still categorised correctly."""
    import numpy as np

    def f(a: In, *, M: Compile[int], dtype: Compile[type] = np.float32):
        pass

    compile_params, tensor_params, scalar_params = _split_params(f)
    assert compile_params == ["M", "dtype"]
    assert tensor_params == ["a"]
    assert scalar_params == []


def test_split_params_scalar_with_default():
    def f(a: In, alpha: float = 1.0, *, N: Compile[int] = 512):
        pass

    compile_params, tensor_params, scalar_params = _split_params(f)
    assert compile_params == ["N"]
    assert tensor_params == ["a"]
    assert scalar_params == ["alpha"]
