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
from aie.utils.compile.jit._introspect import declare
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
    # alias already carries an Annotated origin -- this is why a bare
    # CompileTime below still declares a parameter.
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


def declared_as(annotation) -> str:
    """What ``declare`` makes of one parameter annotated ``annotation``."""

    def generator(x):
        pass

    if annotation is not inspect.Parameter.empty:
        generator.__annotations__ = {"x": annotation}
    d = declare(generator)
    if "x" in d.params:
        return "compile"
    if [m.name for m in d.operands] == ["x"]:
        return "tensor"
    if [m.name for m in d.per_calls] == ["x"]:
        return f"dispatch {d.per_calls[0].dtype!r}"
    assert d.scalars == ("x",)
    return "scalar"


@pytest.mark.parametrize(
    "annotation, kind",
    [
        (CompileTime[int], "compile"),
        (CompileTime[str], "compile"),
        (CompileTime, "compile"),
        (In, "tensor"),
        (Out, "tensor"),
        (InOut, "tensor"),
        (DispatchTime[int], f"dispatch {int!r}"),
        (DispatchTime[np.int32], f"dispatch {np.int32!r}"),
        (int, "scalar"),
        (float, "scalar"),
        (str, "scalar"),
        (None, "scalar"),
        (inspect.Parameter.empty, "scalar"),
    ],
)
def test_declare_sorts_a_parameter_by_its_annotation(annotation, kind):
    assert declared_as(annotation) == kind


def test_a_bare_dispatch_marker_declares_a_per_call_value():
    assert declared_as(DispatchTime).startswith("dispatch ")


@pytest.mark.parametrize(
    "annotation, kind",
    [(CompileTime[int], "compile"), (DispatchTime[int], f"dispatch {int!r}")],
)
def test_parameter_metadata_uses_value_not_identity(annotation, kind):
    metadata = json.loads(json.dumps(get_args(annotation)[1]))
    assert declared_as(Annotated[int, metadata]) == kind
    assert "0x" not in repr(annotation)


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
