# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""Per-call integers: the algebra on the host, and its staging to arith."""

import numpy as np
import pytest
from aie.dialects import func
from aie.extras import types as T
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects.arith import constant
from aie.helpers.taplib import Affine, PerCall, TensorAccessPattern
from aie.ir import IntegerAttr
from aie.iron import ceildiv
from aie.passmanager import PassManager


def per_call(name="p"):
    return PerCall(name, "aie.dispatch_time", np.int32)


def staged(e: Affine, v: int) -> int:
    """``e`` staged on the constant ``v`` and folded."""
    with mlir_mod_ctx() as ctx:

        @func.FuncOp.from_py_func(name="f")
        def _():
            e.value.ssa = constant(v, T.i32())
            return e.stage()

        PassManager.parse("builtin.module(func.func(canonicalize))").run(
            ctx.module.operation
        )
        ret = list(ctx.module.body.operations[0].regions[0].blocks[0].operations)[-1]
        return IntegerAttr(ret.operands[0].owner.attributes["value"]).value


def test_arithmetic_on_a_per_call_value_is_an_expression():
    p = per_call()
    assert (p + 1) * 64 == Affine(p, 64, 64) == 64 * (1 + p)
    assert 10 - p == Affine(p, -1, 10) and -p == Affine(p, -1)
    assert (p * 4) // 2 == p // 1 * 2 and p // 2 == Affine(p, 1, 0, 1)
    exprs = (p.affine(), p * 64, (p + 1) * 64, p - 1, 10 - p)
    assert [e.name for e in exprs] == ["p", "p_x64", "p_x64_p64", "p_m1", "p_xm1_p10"]
    assert [str(e) for e in exprs] == ["p", "p * 64", "p * 64 + 64", "p - 1", "-p + 10"]
    assert str(((p * 3 - 5) // 4) * 8 // 2 - 1) == "((p * 3 - 5) >> 2) * 4 - 1"
    assert p == p and p != per_call() and p + 1 == p + 1
    assert len({p, p}) == 1 and len({p + 1, 1 + p}) == 1
    assert p != "p" and p + 1 != "p" and p != None  # noqa: E711
    for bad in (lambda: p * 0.5, lambda: p < 0, lambda: p + 1 == 1, lambda: bool(p)):
        with pytest.raises(TypeError):
            bad()
    with pytest.raises(TypeError, match="not a number"):
        int(p)


# (f, error) on x = 3 v - 5: f(x) evaluates as f(3 v - 5) at every v, or raises.
_X = Affine(per_call("x"), 3, -5)
_Y = per_call("y")
_CASES = [
    (lambda v: v + 7, None),
    (lambda v: 2 - v, None),
    (lambda v: v * -4, None),
    (lambda v: v // 8, None),
    (lambda v: v // -4, None),
    (lambda v: ceildiv(v, 64), None),
    (lambda v: (ceildiv(v, 64) + 1) * 32, None),
    (lambda v: (v // 4 + 3) // 16, None),
    (lambda v: (v // 4) * 8 // 2 - 1, None),
    (lambda v: (v * 512) // 512, None),
    (lambda v: ceildiv(v * 4, 256), None),
    (lambda v: v * 3 + v + 1, None),
    (lambda v: v * 4 - v, None),
    (lambda v: v - v, None),
    (lambda v: (v * 4) // v, None),
    (lambda v: v % 64, None),
    (lambda v: max(v - 3, 0) if isinstance(v, int) else (v - 3).clamp(lo=0), None),
    (lambda v: v // 3, TypeError),
    (lambda v: (v // 4) * 3 // 2, TypeError),
    (lambda v: v % 48, TypeError),
    (lambda v: v + _Y, TypeError),
    (lambda v: v // 4 + v, TypeError),
    (lambda v: (v + 1) // v, TypeError),
    (lambda v: v // (v * 2), TypeError),
]


@pytest.mark.parametrize("f, error", _CASES)
def test_an_expression_computes_what_it_says(f, error):
    if error is not None:
        with pytest.raises(error):
            f(_X)
        return
    form = f(_X)
    for v in range(-300, 300):
        want = f(3 * v - 5)
        got = form if isinstance(form, int) else form.evaluate({"x": v})
        assert got == want, v


def test_a_clamp_is_carried_through_sums_products_and_floors():
    p = per_call()
    start = (p - 3).clamp(lo=0)
    assert start.name == "p_m3_lo0" and str(start) == "max(p - 3, 0)"
    assert str(start.clamp(hi=6)) == "clamp(p - 3, 0, 6)"
    cases = {
        start + 5: lambda v: max(v - 3, 0) + 5,
        start * -2: lambda v: -2 * max(v - 3, 0),
        (p + 1).clamp(hi=9) // 4: lambda v: min(v + 1, 9) // 4,
        (p - 3).clamp(lo=0).clamp(lo=-5, hi=6): lambda v: min(max(v - 3, 0), 6),
    }
    for e, want in cases.items():
        assert [e.evaluate({"p": v}) for v in range(-20, 21)] == [
            want(v) for v in range(-20, 21)
        ], e
    with pytest.raises(TypeError, match="not linear"):
        start.affine()
    for bad in (lambda: start + start, lambda: start // 3):
        with pytest.raises(TypeError):
            bad()
    with pytest.raises(ValueError, match="holds no value"):
        (p + 0).clamp(lo=5, hi=2)


def test_a_wrapped_expression_is_a_ring_row_and_its_elements():
    p = per_call()
    ring = p % 16
    assert ring.name == "p_mod16" and str(ring * 64) == "(p * 64) % 1024"
    assert (ring * 64).evaluate({"p": 21}) == 5 * 64
    for bad in (lambda: p % 12, lambda: ring % 32, lambda: ring * 3, lambda: ring + 1):
        with pytest.raises(TypeError):
            bad()


@pytest.mark.parametrize(
    "build",
    [
        lambda p: p * 64 + 64,
        lambda p: ceildiv(p * 3 - 5, 64),
        lambda p: ((p * 3 + 1) // 2).clamp(lo=-4, hi=20) * 3 + 1,
        lambda p: (p - 5).clamp(lo=0) % 8 * 4,
    ],
)
def test_a_staged_expression_folds_to_its_host_value(build):
    p = per_call()
    e = build(p)
    for v in (-37, -1, 0, 5, 300):
        assert staged(e, v) == e.evaluate({"p": v}), v


def test_a_pattern_takes_an_expression_and_stages_it():
    p = per_call()
    with mlir_mod_ctx():

        @func.FuncOp.from_py_func(T.i32(), name="f")
        def _(arg):
            p.ssa = arg
            tap = TensorAccessPattern.full((64, 16)).partition(4)[p + 1]
            assert all(isinstance(x, int) for x in tap.sizes)
            assert not isinstance(tap.offset, (int, Affine))


def test_an_unbound_expression_does_not_stage():
    with pytest.raises(TypeError, match="no staged value"):
        (per_call() + 1).stage()
