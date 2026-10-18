# test_declaration.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Declaration: what an operator class declares, its checks and inference — no NPU required."""

from __future__ import annotations

from enum import Enum
from typing import ClassVar

import numpy as np
import pytest
from aie.utils.compile.jit.declaration import Declaration
from aie.utils.compile.jit.markers import In, InOut, Out
from aie.utils.compile.jit.params import (
    Select,
    param,
    per_call,
    tunable,
    widths,
)


class Base:
    trace = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.declaration = Declaration.of_class(cls, Base)

    def inputs(self):
        pass


class Placement(Enum):
    SHIMS = 1
    MEMTILES = 2


class GEMV(Base):
    batches: int | None = param(default=None)
    M: int = param()
    K: int = param()
    b_col_maj: bool = param(default=False)
    dtype: type = param(default=np.int8)
    scale: float = param(default=1.0, array=True)
    rows: int = param(derive=lambda op: op.M * 2)
    tile: int = tunable(default=16, domain=(16, 32))
    cores: int = tunable(default=1, domain=widths)
    placement: Placement = tunable(default=Placement.SHIMS, domain=Placement)
    width: int = tunable(init=False)
    note: ClassVar[str] = "kept"
    a = In(batches, M, K, dtype=dtype, obj_shape=tile, per=cores)
    b = In(Select(b_col_maj, (K, 1), (1, K)))
    c = Out(M)
    valid = per_call(bounds=M)


class Wider(GEMV):
    extra: int = param(default=0)
    c = InOut(GEMV.M)


def test_a_declaration_records_parameters_members_and_the_key():
    d = GEMV.declaration
    assert list(d.params) == [
        "batches",
        "M",
        "K",
        "b_col_maj",
        "dtype",
        "scale",
        "rows",
        "tile",
        "cores",
        "placement",
        "width",
    ]
    assert [m.name for m in d.members] == ["a", "b", "c", "valid"]
    assert d.nullable == {"batches"}
    assert d.tunables == ("tile", "cores", "placement")
    assert d.key == ("dtype", "scale", "tile", "cores", "placement", "width")


def test_a_subclass_inherits_and_its_own_body_orders_first():
    d = Wider.declaration
    assert list(d.params)[-1] == "extra" and d.nullable == {"batches"}
    assert [m.name for m in d.members] == ["c", "a", "b", "valid"]
    assert isinstance(Wider.c, InOut) and Wider.declaration.params["M"] is GEMV.M


def test_a_declaration_prints_what_it_declared():
    assert repr(GEMV.declaration) == "\n".join(
        [
            "GEMV",
            "  params:   batches, M, K, b_col_maj, dtype, scale",
            "  derived:  rows",
            "  tunables: tile in {16, 32}, cores in widths, placement in {SHIMS, MEMTILES}",
            "  resolved: width",
            "  In        a [batches, M, K] per [cores]",
            "  In        b [Select(b_col_maj, (K, 1), (1, K))]",
            "  Out       c [M]",
            "  per_call  valid",
            "  key:      (dtype, scale, tile, cores, placement, width)",
        ]
    )


def declare(body: dict, annotations: dict | None = None, base: type = Base):
    return type("Bad", (base,), {**body, "__annotations__": annotations or {}})


@pytest.mark.parametrize(
    "body, annotations, message",
    [
        ({"a": In(4)}, {"a": "int"}, "members are declared without an annotation"),
        ({"inputs": In(4)}, {}, "hides Base.inputs, which the library reads"),
        ({"M": param()}, {}, "annotate the parameter with its type"),
        ({}, {"M": "int"}, "an annotated attribute is a param()"),
        ({"M": 4}, {"M": "int"}, "an annotated attribute is a param()"),
        ({"M": 4}, {}, "does not override the inherited parameter"),
        (
            {"trace": param()},
            {"trace": "int"},
            "hides Base.trace, which the library reads",
        ),
    ],
)
def test_a_class_body_breaking_a_rule_is_refused(body, annotations, message):
    base = GEMV if message.startswith("does not") else Base
    with pytest.raises(
        TypeError, match=message.replace("(", "\\(").replace(")", "\\)")
    ):
        declare(body, annotations, base)


M, T, F = param(), tunable(), param(default=None)
M.name, T.name, F.name = "M", "T", "F"


@pytest.mark.parametrize(
    "member, message",
    [
        (per_call(bounds=T), "must name a param\\(\\), the one a call may bound"),
        (In(M, True), "dimension True is not a param\\(\\) of Bad or an integer"),
        (In(M, 2.0), "dimension 2.0 is not a param\\(\\) of Bad or an integer"),
        (In(Select(M, (T,), (1,))), "dimension T is a tunable\\(\\)"),
        (In(F, M, F), "at most one T | None dimension"),
        (In(M, when=T), "when=<tunable T> must be a param\\(\\)"),
        (In(M, obj_shape=(2.5,)), "obj_shape dimension 2.5 is not"),
        (In(M, per=3), "per=3 must be a param\\(\\) or tunable\\(\\)"),
        (In(GEMV.M), "dimension <param M> is not a param\\(\\) of Bad"),
    ],
)
def test_a_member_breaking_a_rule_is_refused(member, message):
    annotations = {"M": "int", "T": "int", "F": "int | None"}
    with pytest.raises(TypeError, match=message):
        declare({"M": M, "T": T, "F": F, "x": member}, annotations)


def test_inference_reads_the_parameters_off_the_shapes():
    d = GEMV.declaration
    assert d.infer({"a": (8, 4), "b": (1, 4)}) == {"M": 8, "K": 4, "batches": None}
    assert d.infer({"a": (3, 8, 4), "b": (4, 1)}, b_col_maj=True) == {
        "b_col_maj": True,
        "batches": 3,
        "M": 8,
        "K": 4,
    }
    assert d.infer({"a": (8, 4), "b": (1, 4)}, outputs=[(2, 4)]) == {
        "M": 8,
        "K": 4,
        "batches": None,
    }


def test_inference_refuses_shapes_that_disagree():
    d = GEMV.declaration
    with pytest.raises(ValueError, match="K is 5 from b.shape\\[1\\] but 4 from a"):
        d.infer({"a": (8, 4), "b": (1, 5)})
    with pytest.raises(ValueError, match="K is 4 from a.shape\\[1\\] but 5 from given"):
        d.infer({"a": (8, 4), "b": (1, 4)}, K=5)
    with pytest.raises(ValueError, match="b.shape\\[0\\] is 2, declared 1"):
        d.infer({"a": (8, 4), "b": (2, 4)})
    with pytest.raises(ValueError, match="operand a has rank 4, declared"):
        d.infer({"a": (1, 2, 8, 4), "b": (1, 4)})
    with pytest.raises(ValueError, match="operand b has rank 3"):
        d.infer({"a": (8, 4), "b": (1, 4, 1)})


def test_inference_refuses_the_wrong_operands():
    d = GEMV.declaration
    with pytest.raises(TypeError, match="takes 2 operand\\(s\\) \\(a, b\\), got 1"):
        d.infer({"a": (8, 4)})
    with pytest.raises(TypeError, match="produces 1 output\\(s\\) \\(c\\), got 2"):
        d.infer({"a": (8, 4), "b": (1, 4)}, outputs=[(8,), (8,)])


class Gated(Base):
    M: int = param()
    flip: bool = param()
    biased: bool = param(default=False)
    x = In(M)
    bias = In(M, when=biased)
    scale = In(M, when=biased)
    y = Out(Select(flip, (M, 1), (1, M)))


def test_a_flat_operand_takes_any_rank_and_a_when_operand_sets_its_flag():
    d = Gated.declaration
    assert d.infer({"x": (2, 4)}) == {"biased": False, "M": 8}
    assert d.infer({"x": (4,), "bias": (4,), "scale": (4,)}, flip=False) == {
        "flip": False,
        "biased": True,
        "M": 4,
    }


def test_inference_refuses_a_flag_its_operands_contradict():
    d = Gated.declaration
    with pytest.raises(
        TypeError, match="the operands biased= brings are given together"
    ):
        d.infer({"x": (4,), "bias": (4,)})
    with pytest.raises(TypeError, match="Gated\\(biased=True\\) takes bias="):
        d.infer({"x": (4,)}, biased=True)
    with pytest.raises(
        TypeError, match="bias= is an operand only where biased is true"
    ):
        d.infer({"x": (4,), "bias": (4,), "scale": (4,)}, biased=False)


def test_a_select_with_no_flag_value_asks_for_it():
    with pytest.raises(ValueError, match="flip selects y's shape and has no default"):
        Gated.declaration.infer({"x": (4,)}, outputs=[(4, 1)])


class Derived(Base):
    M: int = param()
    flagged: bool = param(default=lambda op: True)
    x = In(M)
    y = In(M, when=flagged)


def test_a_flag_a_function_computes_is_absent_until_given():
    assert Derived.declaration.infer({"x": (4,)}) == {"flagged": False, "M": 4}


class Narrowed(GEMV):
    batches: int = param(default=1)


def test_a_subclass_may_make_a_nullable_parameter_required_and_dims_may_be_numpy():
    assert Narrowed.declaration.nullable == frozenset()
    assert declare({"M": M, "x": In(M, np.int64(2))}, {"M": "int"}).declaration.params
