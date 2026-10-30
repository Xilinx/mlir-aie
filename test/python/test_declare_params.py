# test_declare_params.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""param(), tunable(), per_call(), Select, the domain functions and the In/Out/InOut operand — no NPU required."""

from __future__ import annotations

import functools

import numpy as np
import pytest
from aie.iron.device import NPU2Col4
from aie.utils import coded
from aie.utils.compile.jit.markers import In, InOut, Out
from aie.utils.compile.jit.operator import Operator
from aie.utils.compile.jit.params import (
    Select,
    divisors,
    param,
    per_call,
    tunable,
    widths,
)
from ml_dtypes import bfloat16


class Op:
    M = param()
    tile = tunable(default=64, domain=(32, 64))
    count = per_call(derive=lambda op: op.M // 4)
    valid = per_call(bounds=M)
    edges = per_call(count=4, derive=lambda op: tuple(i * op.M for i in range(4)))
    b_col_maj = param(default=False, array=True)
    shape = Select(b_col_maj, (M, 1), (1, M))


def test_a_parameter_is_its_own_reference_and_an_instance_holds_its_value():
    op = Op()
    assert Op.M.name == "M" and Op.M.owner is Op
    vars(op)["M"] = 8
    assert op.M == 8


def test_a_parameter_takes_a_default_or_a_derive_not_both():
    with pytest.raises(TypeError, match="give one, not both"):
        param(default=1, derive=lambda op: 1)


def test_widths_are_the_start_and_each_power_of_two_up_to_the_columns():
    assert widths(None, NPU2Col4(), 3) == (4, 3, 2, 1)


def test_a_domain_is_a_tuple_an_enum_or_a_function():
    with pytest.raises(TypeError, match="give a tuple, an Enum or a function"):
        tunable(domain=[1, 2])
    assert tunable(domain=functools.partial(widths)).domain is not None


def test_a_mutable_default_is_refused():
    with pytest.raises(TypeError, match="would be shared by every instance"):
        param(default=[])


def test_divisors_double_and_halve_within_their_bounds():
    assert divisors(64, of=256) == (256, 128, 64, 32, 16, 8, 4, 2, 1)
    assert divisors(64, of=256, step=16, cap=128) == (128, 64, 32, 16)
    assert divisors(64, of=256, span=1) == (128, 64, 32)


def test_a_per_call_value_is_an_integer_with_one_source():
    with pytest.raises(TypeError, match="a per-call value is an integer"):
        per_call(np.float32)
    with pytest.raises(TypeError, match="give one, not both"):
        per_call(derive=lambda op: 1, bounds=Op.M)


def test_a_per_call_vector_is_a_fixed_word_count():
    op = Op()
    vars(op)["M"] = 8
    assert Op.edges.count == 4
    assert Op.edges.derive(op) == (0, 8, 16, 24)
    counted = per_call(count=lambda op: op.M // 2)
    assert counted.count(op) == 4


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(count=4, bounds=Op.M), "a vector bounds no single param"),
        (dict(count=0), "give a positive int or a function"),
        (dict(count=-1), "give a positive int or a function"),
        (dict(count=True), "give a positive int or a function"),
    ],
)
def test_a_per_call_vector_rejects_bounds_and_a_non_positive_count(kwargs, message):
    with pytest.raises(TypeError, match=message):
        per_call(**kwargs)


def test_a_select_picks_by_its_flag():
    op = Op()
    vars(op).update(b_col_maj=True)
    assert Op.shape.branch(op) == (Op.M, 1)
    vars(op).update(b_col_maj=None)
    with pytest.raises(ValueError, match="needs a resolved operator"):
        Op.shape.branch(op)


class Streams(Operator):
    M: int = param()
    batches: int | None = param(default=None)
    K: int = param(default=4)
    dtype: type = param(default=np.int8)
    tile: int = tunable()
    cores: int = tunable(default=2)
    gated: bool = param(default=False)
    a = In(batches, M, K, dtype=dtype, obj_shape=tile, per=(cores, K))
    b = Out(Select(gated, (K,), (M, K)), replicate=True, per=cores, when=gated)
    c = InOut(M)


def streams(**values):
    op = Streams(M=8, tile=16)
    vars(op).update(values)
    return op


def test_an_operand_resolves_its_shapes_on_the_operator():
    a = Streams.a.of(streams())
    assert (a.shape, a.elements, a.dtype) == ((8, 4), 32, np.int8)
    assert a.streamed and a.obj_shape == (16,) and a.lanes == 8
    assert Streams.a.of(streams(batches=3)).shape == (3, 8, 4)
    assert Streams.b.of(streams(gated=True)).shape == (4,)
    assert not Streams.c.streamed and Streams.c.of(streams()).shape == (8,)


class Coded(Operator):
    M: int = param()
    K: int = param()
    rows: int = param(default=4)
    coded_w: bool = tunable(default=False, domain=(False, True))
    w = In(M, K, obj_shape=(rows, K), per=(rows,), coded=coded_w)


def test_a_coded_operand_moves_its_rows_as_bytes_and_keeps_its_logical_shape():
    plain = Coded.w.of(Coded(M=8, K=2048))
    assert (plain.shape, plain.dtype, plain.obj_shape) == (
        (8, 2048),
        bfloat16,
        (4, 2048),
    )
    w = Coded.w.of(Coded(M=8, K=2048, coded_w=True))
    R = coded.row_bytes(2048)
    assert (w.logical_shape, w.logical_dtype) == ((8, 2048), bfloat16)
    assert (w.shape, w.dtype, w.obj_shape, w.elements) == (
        (8, R),
        np.uint8,
        (4, R),
        8 * R,
    )
    assert w.ndarray_type == np.ndarray[(8, R), np.dtype[np.uint8]]
    assert repr(w) == f"<In w (8, {R}) uint8 coded>"
    assert "coded_w" in Coded.declaration.key


def test_a_coded_operand_refuses_what_its_rows_cannot_hold():
    class Narrow(Operator):
        K: int = param()
        x = In(K, 2, dtype=np.float32, obj_shape=(K, 2), coded=True)
        y = In(2, K, obj_shape=(128,), coded=True)

    op = Narrow(K=64)
    with pytest.raises(TypeError, match="coded rows hold bfloat16"):
        Narrow.x.of(op).dtype
    with pytest.raises(ValueError, match="whole rows of 64"):
        Narrow.y.of(op).obj_shape
    with pytest.raises(TypeError, match="coded= takes a bool"):

        class Wrong(Operator):
            K: int = param()
            x = In(K, coded="yes")


def test_an_operand_says_which_way_it_moves():
    assert (In.fills, In.drains, Out.fills, Out.drains) == (True, False, False, True)
    assert InOut.fills and InOut.drains


def test_an_operand_prints_its_declaration_and_its_resolved_shape():
    assert repr(Streams.a) == "In(batches, M, K)"
    assert repr(Streams.a.of(streams())) == "<In a (8, 4) int8>"


def test_a_replicated_operand_needs_a_split():
    with pytest.raises(TypeError, match="replicate=True needs per="):
        In(Op.M, replicate=True)


@pytest.mark.parametrize("stream_when_true", [False, True])
def test_a_select_obj_shape_streams_only_where_its_flag_picks_a_shape(
    stream_when_true,
):
    class MaybeStreamed(Operator):
        K: int = param()
        fed: bool = tunable(default=False, domain=(False, True))
        x = In(
            K,
            obj_shape=Select(
                fed,
                (K,) if stream_when_true else None,
                None if stream_when_true else (K,),
            ),
        )

    assert MaybeStreamed.x.streamed
    off = MaybeStreamed.x.of(MaybeStreamed(K=64, fed=not stream_when_true))
    assert not off.streamed
    with pytest.raises(TypeError, match="x has no obj_shape here: no stream"):
        off.lanes
    with pytest.raises(TypeError, match="x has no obj_shape here: no stream"):
        off.obj_shape
    on = MaybeStreamed.x.of(MaybeStreamed(K=64, fed=stream_when_true))
    assert on.streamed and on.obj_shape == (64,) and on.lanes == 1


def test_a_select_of_two_shapes_streams_before_its_flag_resolves():
    class Tiled(Operator):
        K: int = param()
        wide: bool = tunable(default=False, domain=(False, True))
        cores: int = tunable(default=2)
        x = In(K, obj_shape=Select(wide, (K,), (1,)), per=cores)

    op = Tiled(K=64)
    vars(op).update(wide=None)
    x = Tiled.x.of(op)
    assert x.streamed and x.lanes == 2


def test_an_unstreamed_operand_has_no_element_or_lanes():
    c = Streams.c.of(streams())
    with pytest.raises(TypeError, match="c has no obj_shape here: no stream"):
        c.obj_shape
    with pytest.raises(TypeError, match="c has no obj_shape here: no stream"):
        c.lanes


def test_a_dimension_that_is_none_and_not_nullable_is_refused():
    with pytest.raises(ValueError, match="tile is None: resolve the operator first"):
        Streams.a.of(streams(tile=None)).obj_shape
    with pytest.raises(ValueError, match="M is None; it must be set"):
        Streams.c.of(streams(M=None)).shape


def test_an_absent_operand_names_its_flag():
    with pytest.raises(AttributeError, match="declared when=gated, which is False"):
        streams().b
