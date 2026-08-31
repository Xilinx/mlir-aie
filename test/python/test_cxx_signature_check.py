# test_cxx_signature_check.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Checking a kernel's `arg_types` against its real C++ signature.

Only possible for C++-linkage kernels: an ``extern "C"`` symbol demangles to a
bare name with no parameter list, so there is nothing to compare against.
Dropping the trampolines is what buys the check.

Three tiers of strictness, by how much a false positive would cost:
  arity           -- hard error, the parameter count is unambiguous
  pointer/scalar  -- hard error, equally unambiguous
  element type    -- checked only when both sides are understood, so kernels
                     taking aie_api vectors or structs are never rejected
"""

import numpy as np
import pytest
from ml_dtypes import bfloat16

from aie.utils.compile.utils import (
    _check_cxx_signature,
    _normalize_cxx_param,
    _split_top_level_params,
)

I32_ARRAY = np.ndarray[(1024,), np.dtype[np.int32]]
F32_ARRAY = np.ndarray[(1024,), np.dtype[np.float32]]
BF16_ARRAY = np.ndarray[(1024,), np.dtype[bfloat16]]

# ---------------------------------------------------------------------------
# Splitting a demangled parameter list.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params,expected",
    [
        ("", []),
        ("int", ["int"]),
        ("int*, int*, int", ["int*", "int*", "int"]),
        # A template argument list contains commas that are NOT parameter
        # separators -- splitting naively would report 3 parameters here.
        ("aie::vector<int, 16>*, int", ["aie::vector<int, 16>*", "int"]),
        # So does a function-pointer parameter.
        ("void (*)(int, int), float", ["void (*)(int, int)", "float"]),
    ],
)
def test_split_respects_nesting(params, expected):
    assert _split_top_level_params(params) == expected


# ---------------------------------------------------------------------------
# Normalizing one parameter to (base type, is_pointer).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "param,base,is_pointer",
    [
        ("int", "int", False),
        ("int*", "int", True),
        # llvm-cxxfilt emits east const.
        ("int const*", "int", True),
        ("const int*", "int", True),
        # Top-level restrict is dropped by the mangling itself, but be robust.
        ("int* __restrict", "int", True),
        ("unsigned char*", "unsigned char", True),
        ("bfloat16*", "bfloat16", True),
        # A reference binds like a pointer for our purposes.
        ("int&", "int", True),
    ],
)
def test_normalize_param(param, base, is_pointer):
    assert _normalize_cxx_param(param) == (base, is_pointer)


# ---------------------------------------------------------------------------
# Tier 1: arity.
# ---------------------------------------------------------------------------


def test_matching_signature_is_accepted():
    _check_cxx_signature(
        "reduce_min_vector(int*, int*, int)",
        [I32_ARRAY, I32_ARRAY, np.int32],
        "reduce_min_vector",
    )


def test_too_few_declared_arguments_is_an_error():
    with pytest.raises(ValueError) as exc:
        _check_cxx_signature("f(int*, int*, int)", [I32_ARRAY, np.int32], "f")
    message = str(exc.value)
    assert "3" in message and "2" in message
    assert "f(int*, int*, int)" in message


def test_too_many_declared_arguments_is_an_error():
    with pytest.raises(ValueError) as exc:
        _check_cxx_signature("f(int*)", [I32_ARRAY, np.int32], "f")
    assert "f(int*)" in str(exc.value)


def test_empty_parameter_list_is_handled():
    _check_cxx_signature("f()", [], "f")


# ---------------------------------------------------------------------------
# Tier 2: pointer vs scalar.
# ---------------------------------------------------------------------------


def test_buffer_passed_where_scalar_expected_is_an_error():
    with pytest.raises(ValueError) as exc:
        _check_cxx_signature("f(int*, int)", [I32_ARRAY, I32_ARRAY], "f")
    message = str(exc.value)
    assert "argument 2" in message.lower()


def test_scalar_passed_where_buffer_expected_is_an_error():
    with pytest.raises(ValueError) as exc:
        _check_cxx_signature("f(int*, int)", [np.int32, np.int32], "f")
    assert "argument 1" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Tier 3: element type, checked only when both sides are understood.
# ---------------------------------------------------------------------------


def test_element_type_mismatch_is_an_error():
    with pytest.raises(ValueError) as exc:
        _check_cxx_signature("f(float*, int)", [I32_ARRAY, np.int32], "f")
    message = str(exc.value)
    assert "float" in message
    assert "int32" in message


def test_const_and_restrict_do_not_affect_the_comparison():
    _check_cxx_signature(
        "f(int const*, int* __restrict, int)",
        [I32_ARRAY, I32_ARRAY, np.int32],
        "f",
    )


@pytest.mark.parametrize(
    "cxx,dtype",
    [
        ("signed char", np.int8),
        ("unsigned char", np.uint8),
        ("short", np.int16),
        ("unsigned short", np.uint16),
        ("int", np.int32),
        ("unsigned int", np.uint32),
        ("float", np.float32),
        ("bfloat16", bfloat16),
    ],
)
def test_known_element_types_are_accepted(cxx, dtype):
    array = np.ndarray[(64,), np.dtype[dtype]]
    _check_cxx_signature(f"f({cxx}*)", [array], "f")


def test_unknown_cxx_type_is_not_second_guessed():
    """A kernel taking an aie_api vector demangles to a spelling we do not
    model.  Staying silent is the point: a checker that does not understand a
    type must not reject it."""
    _check_cxx_signature("f(aie::vector<int, 16>*, int)", [I32_ARRAY, np.int32], "f")


def test_unknown_numpy_dtype_is_not_second_guessed():
    _check_cxx_signature("f(int*)", [np.ndarray[(64,), np.dtype[np.complex64]]], "f")


def test_bfloat16_buffer_against_float_kernel_is_an_error():
    """bf16 and f32 buffers are the same pointer at the ABI level but half the
    element width -- exactly the mismatch that silently corrupts today."""
    with pytest.raises(ValueError) as exc:
        _check_cxx_signature("f(float*)", [BF16_ARRAY], "f")
    assert "bfloat16" in str(exc.value)


def test_float_buffer_against_bfloat16_kernel_is_an_error():
    with pytest.raises(ValueError) as exc:
        _check_cxx_signature("f(bfloat16*)", [F32_ARRAY], "f")
    assert "float32" in str(exc.value)


def test_missing_arg_types_skips_the_check():
    """Kernels constructed without a signature cannot be checked."""
    _check_cxx_signature("f(int*, int)", None, "f")
