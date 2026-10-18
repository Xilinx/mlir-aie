# test_declare_params.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""param(), tunable(), per_call(), Select, the domain functions and the In/Out/InOut operand — no NPU required."""

from enum import Enum
from types import SimpleNamespace

import numpy as np
import pytest
from aie.utils.compile.jit.markers import In, InOut, Out
from aie.utils.compile.jit.params import (
    Select,
    divisors,
    param,
    per_call,
    tunable,
    widths,
)
from ml_dtypes import bfloat16


class Placement(Enum):
    SHIMS = 1
    MEMTILES = 2


class Op:
    M = param()
    rows = param(default=lambda op: op.M * 2)
    dtype = param(default=bfloat16)
    pad = param(derive=lambda op: op.M + 1)
    tile = tunable(default=64, domain=(32, 64))
    width = tunable(domain=widths)
    placement = tunable(default=Placement.SHIMS, domain=Placement)
    count = per_call(derive=lambda op: op.M // 4)
    valid = per_call(bounds=M)
    b_col_maj = param(default=False, array=True)
    shape = Select(b_col_maj, (M, 1), (1, M))


def test_a_parameter_is_its_own_reference_and_an_instance_holds_its_value():
    op = Op()
    assert Op.M.name == "M" and Op.M.owner is Op
    vars(op)["M"] = 8
    assert op.M == 8


def test_a_parameter_an_instance_does_not_hold_names_the_layout_key():
    with pytest.raises(AttributeError, match="build\\(\\) reads M, which is not in"):
        Op().M


def test_a_function_default_is_computed_and_any_other_is_the_value():
    op = Op()
    vars(op)["M"] = 8
    assert Op.rows.initial(op) == 16
    assert Op.dtype.initial(op) is bfloat16
    assert Op.pad.initial(op) == 9


def test_a_parameter_takes_a_default_or_a_derive_not_both():
    with pytest.raises(TypeError, match="give one, not both"):
        param(default=1, derive=lambda op: 1)


def test_a_tunable_is_always_keyed_and_a_parameter_only_when_marked():
    assert Op.tile.array and Op.tile.tunable
    assert not Op.M.array and Op.b_col_maj.array


def test_a_tuple_domain_puts_a_start_outside_it_first():
    assert Op.tile.values(None, None, 64) == (32, 64)
    assert Op.tile.values(None, None, 16) == (16, 32, 64)


def test_an_enum_domain_puts_the_start_first():
    assert Op.placement.values(None, None, Placement.MEMTILES) == (
        Placement.MEMTILES,
        Placement.SHIMS,
    )


def test_a_function_domain_is_called_and_none_is_not_searched():
    dev = SimpleNamespace(cols=4)
    assert Op.width.values(None, dev, 3) == (4, 3, 2, 1)
    assert Op.M.values(None, dev, 8) == ()


def test_a_domain_is_a_tuple_an_enum_or_a_function():
    with pytest.raises(TypeError, match="give a tuple, an Enum or a function"):
        tunable(domain=[1, 2])


def test_divisors_double_and_halve_within_their_bounds():
    assert divisors(64, of=256) == (256, 128, 64, 32, 16, 8, 4, 2, 1)
    assert divisors(64, of=256, step=16, cap=128) == (128, 64, 32, 16)
    assert divisors(64, of=256, span=1) == (128, 64, 32)


def test_a_per_call_value_reads_what_the_operator_bound():
    op = Op()
    op._bound = {"count": 3}
    assert op.count == 3
    assert Op.valid.bounds is Op.M and Op.count.dtype is np.int32
    with pytest.raises(AttributeError, match="Op.valid is not bound yet"):
        op.valid


def test_a_per_call_value_is_an_integer_with_one_source():
    with pytest.raises(TypeError, match="a per-call value is an integer"):
        per_call(np.float32)
    with pytest.raises(TypeError, match="give one, not both"):
        per_call(derive=lambda op: 1, bounds=Op.M)


def test_a_select_picks_by_its_flag():
    op = Op()
    vars(op).update(b_col_maj=True)
    assert Op.shape.branch(op) == (Op.M, 1)
    vars(op).update(b_col_maj=None)
    with pytest.raises(ValueError, match="needs a resolved operator"):
        Op.shape.branch(op)


class Streams:
    M = param()
    batches = param(default=None)
    K = param(default=4)
    dtype = param(default=np.int8)
    tile = tunable()
    cores = tunable(default=2)
    gated = param(default=False)
    a = In(batches, M, K, dtype=dtype, obj_shape=tile, per=(cores, K))
    b = Out(Select(gated, (K,), (M, K)), replicate=True, per=cores, when=gated)
    c = InOut(M)


def streams(**values):
    op = Streams()
    vars(op).update(
        M=8, batches=None, K=4, dtype=np.int8, tile=16, cores=2, gated=False
    )
    vars(op).update(values)
    return op


def test_an_operand_resolves_its_shapes_on_the_operator():
    a = Streams.a.of(streams())
    assert (a.shape, a.elements, a.dtype) == ((8, 4), 32, np.int8)
    assert a.streamed and a.obj_shape == (16,) and a.lanes == 8
    assert Streams.a.of(streams(batches=3)).shape == (3, 8, 4)
    assert Streams.b.of(streams(gated=True)).shape == (4,)
    assert not Streams.c.streamed and Streams.c.of(streams()).shape == (8,)


def test_an_operand_says_which_way_it_moves():
    assert (In.fills, In.drains, Out.fills, Out.drains) == (True, False, False, True)
    assert InOut.fills and InOut.drains


def test_an_operand_prints_its_declaration_and_its_resolved_shape():
    assert repr(Streams.a) == "In(batches, M, K)"
    assert repr(Streams.a.of(streams())) == "<In a (8, 4) int8>"


def test_a_replicated_operand_needs_a_split():
    with pytest.raises(TypeError, match="replicate=True needs per="):
        In(Op.M, replicate=True)


def test_an_unstreamed_operand_has_no_element_or_lanes():
    c = Streams.c.of(streams())
    with pytest.raises(TypeError, match="c is declared without an obj_shape="):
        c.obj_shape
    with pytest.raises(TypeError, match="c is declared without an obj_shape="):
        c.lanes


def test_an_operand_sized_by_an_unresolved_tunable_says_resolve_first():
    with pytest.raises(ValueError, match="tile is None: resolve the operator first"):
        Streams.a.of(streams(tile=None)).obj_shape


def test_an_absent_operand_names_its_flag():
    with pytest.raises(AttributeError, match="declared when=gated, which is False"):
        streams().b
