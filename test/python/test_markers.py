# test_markers.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Unit tests for CompileTime[T], In, Out, InOut, DispatchTime[T] annotation markers — no NPU required."""

import inspect
import json
from typing import TYPE_CHECKING, Annotated, get_args, get_origin

import numpy as np
import pytest
from aie.utils.compile.jit._introspect import (
    _dispatch_param_type,
    _is_compile_param,
    _is_tensor_param,
    declare,
)
from aie.utils.compile.jit.markers import CompileTime, DispatchTime, In, InOut, Out

if TYPE_CHECKING:
    from collections.abc import Sequence

# ---------------------------------------------------------------------------
# CompileTime[T] — Annotated[T, ...] parameterisation
#
# CompileTime[T] is Annotated[T, ...] so pyright/mypy check callers against T
# itself; get_origin/get_args below reflect that, not a bespoke generic.
# ---------------------------------------------------------------------------


def test_compile_int_origin_is_annotated():
    assert get_origin(CompileTime[int]) is Annotated


def test_compile_str_origin_is_annotated():
    assert get_origin(CompileTime[str]) is Annotated


def test_compile_float_origin_is_annotated():
    assert get_origin(CompileTime[float]) is Annotated


def test_compile_type_arg_preserved():
    assert get_args(CompileTime[int])[0] is int
    assert get_args(CompileTime[str])[0] is str


def test_bare_compile_has_annotated_origin():
    # Unlike a bespoke Generic[T], a bare (unsubscripted) Annotated[T, ...]
    # alias already carries an Annotated origin -- this is why
    # _is_compile_param(CompileTime) below is still True.
    assert get_origin(CompileTime) is Annotated


# ---------------------------------------------------------------------------
# DispatchTime[T] — Annotated[T, ...] parameterisation (mirrors CompileTime[T])
# ---------------------------------------------------------------------------


def test_dispatch_int_origin_is_annotated():
    assert get_origin(DispatchTime[int]) is Annotated


def test_dispatch_type_arg_preserved():
    assert get_args(DispatchTime[int])[0] is int
    assert get_args(DispatchTime[str])[0] is str


def test_bare_dispatch_has_annotated_origin():
    assert get_origin(DispatchTime) is Annotated


def test_dispatch_is_not_compile():
    assert DispatchTime is not CompileTime
    assert DispatchTime[int] != CompileTime[int]


@pytest.mark.parametrize(
    "annotation,predicate",
    [
        (CompileTime[int], _is_compile_param),
        (DispatchTime[int], lambda a: _dispatch_param_type(a) is not None),
    ],
)
def test_parameter_metadata_uses_value_not_identity(annotation, predicate):
    metadata = json.loads(json.dumps(get_args(annotation)[1]))
    assert predicate(Annotated[int, metadata])
    assert "0x" not in repr(annotation)


# ---------------------------------------------------------------------------
# _is_compile_param
# ---------------------------------------------------------------------------


def test_is_compile_param_with_int():
    assert _is_compile_param(CompileTime[int]) is True


def test_is_compile_param_with_str():
    assert _is_compile_param(CompileTime[str]) is True


def test_is_compile_param_bare():
    assert _is_compile_param(CompileTime) is True


def test_is_compile_param_rejects_in():
    assert _is_compile_param(In) is False


def test_is_compile_param_rejects_out():
    assert _is_compile_param(Out) is False


def test_is_compile_param_rejects_inout():
    assert _is_compile_param(InOut) is False


def test_is_compile_param_rejects_dispatch():
    assert _is_compile_param(DispatchTime[int]) is False
    assert _is_compile_param(DispatchTime) is False


def test_is_compile_param_rejects_builtin_types():
    assert _is_compile_param(int) is False
    assert _is_compile_param(float) is False
    assert _is_compile_param(str) is False


def test_is_compile_param_rejects_none():
    assert _is_compile_param(None) is False


def test_is_compile_param_rejects_empty():
    assert _is_compile_param(inspect.Parameter.empty) is False


# ---------------------------------------------------------------------------
# _dispatch_param_type
# ---------------------------------------------------------------------------


def test_dispatch_param_type_extracts_int():
    assert _dispatch_param_type(DispatchTime[int]) is int


def test_dispatch_param_type_extracts_numpy_dtype():
    assert _dispatch_param_type(DispatchTime[np.int32]) is np.int32


def test_dispatch_param_type_none_for_non_dispatch():
    assert _dispatch_param_type(DispatchTime) is not None
    for annotation in (CompileTime[int], CompileTime, In, Out, InOut, int, str, None):
        assert _dispatch_param_type(annotation) is None
    assert _dispatch_param_type(inspect.Parameter.empty) is None


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
    assert _is_tensor_param(CompileTime[int]) is False
    assert _is_tensor_param(CompileTime) is False


def test_is_tensor_param_rejects_dispatch():
    assert _is_tensor_param(DispatchTime[int]) is False
    assert _is_tensor_param(DispatchTime) is False


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
    assert In is not CompileTime
    assert Out is not CompileTime
    assert InOut is not CompileTime


def test_tensor_markers_are_not_dispatch():
    assert In is not DispatchTime
    assert Out is not DispatchTime
    assert InOut is not DispatchTime


# ---------------------------------------------------------------------------
# declare() — a generator's parameters by kind, in signature order
# ---------------------------------------------------------------------------


def _compile_only(*, M: CompileTime[int], K: CompileTime[int]):
    pass


def _tensors_only(a: In, b: Out, c: InOut):
    pass


def _dispatch_only(x: DispatchTime[int], y: DispatchTime[float]):
    pass


def _scalars_annotated(x: int, y: float, z: str):
    pass


def _scalars_unannotated(x, y, z):
    pass


def _no_params():
    pass


def _three_kinds(
    a: In, b: Out, alpha: float, *, M: CompileTime[int], N: CompileTime[int]
):
    pass


def _four_kinds(
    a: In,
    b: Out,
    scale: DispatchTime[int],
    alpha: float,
    *,
    M: CompileTime[int],
    N: CompileTime[int],
):
    pass


def _tensors_out_of_order(c: Out, a: In, b: InOut):
    pass


def _compile_out_of_order(
    *, N: CompileTime[int], M: CompileTime[int], K: CompileTime[int]
):
    pass


def _dispatch_out_of_order(
    z: DispatchTime[int], y: DispatchTime[int], x: DispatchTime[int]
):
    pass


def _compile_with_default(
    a: In, *, M: CompileTime[int], dtype: CompileTime[type] = np.float32
):
    pass


def _scalar_with_default(a: In, alpha: float = 1.0, *, N: CompileTime[int] = 512):
    pass


# DispatchTime[T] params are exempt from Guard 1-C's default-value ban.
def _dispatch_with_default(
    a: In, scale: DispatchTime[int] = 1, *, N: CompileTime[int] = 512
):
    pass


@pytest.mark.parametrize(
    "generator, compile_params, tensors, dispatch, scalars",
    [
        (_compile_only, ["M", "K"], [], [], []),
        (_tensors_only, [], ["a", "b", "c"], [], []),
        (_dispatch_only, [], [], ["x", "y"], []),
        (_scalars_annotated, [], [], [], ["x", "y", "z"]),
        (_scalars_unannotated, [], [], [], ["x", "y", "z"]),
        (_no_params, [], [], [], []),
        (_three_kinds, ["M", "N"], ["a", "b"], [], ["alpha"]),
        (_four_kinds, ["M", "N"], ["a", "b"], ["scale"], ["alpha"]),
        (_tensors_out_of_order, [], ["c", "a", "b"], [], []),
        (_compile_out_of_order, ["N", "M", "K"], [], [], []),
        (_dispatch_out_of_order, [], [], ["z", "y", "x"], []),
        (_compile_with_default, ["M", "dtype"], ["a"], [], []),
        (_scalar_with_default, ["N"], ["a"], [], ["alpha"]),
        (_dispatch_with_default, ["N"], ["a"], ["scale"], []),
    ],
)
def test_declare_sorts_a_generators_parameters(
    generator, compile_params, tensors, dispatch, scalars
):
    d = declare(generator)
    assert list(d.params) == compile_params
    assert [m.name for m in d.operands] == tensors
    assert [m.name for m in d.per_calls] == dispatch
    assert list(d.scalars) == scalars


def _checker_only_scalar(
    a: "In", b: "Out", *, M: "CompileTime[int]", cfg: "Sequence[int] | None" = None
):
    pass


def _checker_only_marker(a: "In", *, M: "CompileTime[Sequence[int]]"):
    pass


def test_declare_keeps_markers_beside_a_checker_only_annotation():
    d = declare(_checker_only_scalar)
    assert list(d.params) == ["M"]
    assert [m.name for m in d.operands] == ["a", "b"]
    assert list(d.scalars) == ["cfg"]


def test_declare_names_the_parameter_whose_marker_does_not_resolve():
    with pytest.raises(TypeError, match="Parameter 'M'.*CompileTime\\[Sequence"):
        declare(_checker_only_marker)
