# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Drive every staged branch of the TensorAccessPattern algebra without MLIR.

``Sym`` is an expression-tree stand-in for a runtime scalar: it satisfies the
``__aie_symbolic__`` protocol in ``aie.helpers.taplib.symbolic``, overloads the
integer operators the algebra uses, records the ``select`` and ``require``
decisions the helpers make, and can be evaluated with concrete values. Each
test builds a tiling on ``Sym`` shapes and a ``Sym`` step, evaluates the
resulting offset/sizes/strides for a grid of concrete values, and checks them
against the algebra run on those concrete values directly. That proves the
staged code path computes the same numbers as the integer path, which is the
property the dynamic runtime-sequence builder relies on.
"""

import itertools

import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.helpers.taplib.symbolic import (
    is_sym,
    require,
    sceildiv,
    smin,
    sprod,
    sselect,
)
from util import construct_test

# RUN: %python %s | FileCheck %s


REQUIRED: list = []


class Sym:
    """A recorded integer expression over named runtime scalars."""

    __aie_symbolic__ = True
    __slots__ = ("op", "args")

    def __init__(self, op, *args):
        self.op = op
        self.args = args

    @staticmethod
    def var(name):
        return Sym("var", name)

    @staticmethod
    def _lift(v):
        return v if isinstance(v, Sym) else Sym("const", int(v))

    def _bin(self, op, other, rev=False):
        other = self._lift(other)
        return Sym(op, other, self) if rev else Sym(op, self, other)

    def __add__(self, o):
        return self._bin("add", o)

    def __radd__(self, o):
        return self._bin("add", o, True)

    def __sub__(self, o):
        return self._bin("sub", o)

    def __rsub__(self, o):
        return self._bin("sub", o, True)

    def __mul__(self, o):
        return self._bin("mul", o)

    def __rmul__(self, o):
        return self._bin("mul", o, True)

    def __floordiv__(self, o):
        return self._bin("div", o)

    def __rfloordiv__(self, o):
        return self._bin("div", o, True)

    def __mod__(self, o):
        return self._bin("rem", o)

    def __rmod__(self, o):
        return self._bin("rem", o, True)

    def __lt__(self, o):
        return self._bin("lt", o)

    def __le__(self, o):
        return self._bin("le", o)

    def __gt__(self, o):
        return self._bin("gt", o)

    def __ge__(self, o):
        return self._bin("ge", o)

    def __eq__(self, o):
        return self._bin("eq", o)

    def __ne__(self, o):
        return self._bin("ne", o)

    __hash__ = object.__hash__

    def _select(self, a, b):
        return Sym("select", self, self._lift(a), self._lift(b))

    def _require(self, message):
        REQUIRED.append((self, message))

    def __bool__(self):
        raise TypeError("a Sym has no truth value at generation time")

    __index__ = __int__ = __bool__

    def eval(self, env):
        """Evaluate with Python-int semantics on non-negative operands."""
        if self.op == "var":
            return env[self.args[0]]
        if self.op == "const":
            return self.args[0]
        if self.op == "select":
            c, a, b = (x.eval(env) for x in self.args)
            return a if c else b
        a, b = (x.eval(env) for x in self.args)
        return {
            "add": lambda: a + b,
            "sub": lambda: a - b,
            "mul": lambda: a * b,
            "div": lambda: a // b,
            "rem": lambda: a % b,
            "lt": lambda: a < b,
            "le": lambda: a <= b,
            "gt": lambda: a > b,
            "ge": lambda: a >= b,
            "eq": lambda: a == b,
            "ne": lambda: a != b,
        }[self.op]()

    def __repr__(self):
        return f"Sym({self.op}, {', '.join(map(repr, self.args))})"


def ev(v, env):
    return v.eval(env) if isinstance(v, Sym) else int(v)


def evaluated_tap(tap, env):
    """Return the concrete TensorAccessPattern a staged TensorAccessPattern denotes under ``env``."""
    return TensorAccessPattern(
        [ev(d, env) for d in tap.tensor_dims],
        ev(tap.offset, env),
        [ev(s, env) for s in tap.sizes],
        [ev(s, env) for s in tap.strides],
    )


def check_requires(env, expect_ok):
    """Every recorded guard evaluates to ``expect_ok`` under ``env`` (all of them when ok)."""
    results = [cond.eval(env) for cond, _ in REQUIRED]
    if expect_ok:
        assert all(results), [m for (_, m), r in zip(REQUIRED, results) if not r]
    else:
        assert not all(results)


# CHECK-LABEL: helpers_stage
@construct_test
def helpers_stage():
    a, b = Sym.var("a"), Sym.var("b")
    assert is_sym(a) and not is_sym(3)
    m = smin(a, b)
    assert m.op == "select"
    c = sceildiv(a, b)
    p = sprod([a, 3, b])
    s = sselect(a > b, a, 7)
    REQUIRED.clear()
    require(a % 4 == 0, "a must be a multiple of 4")
    assert len(REQUIRED) == 1
    for env in ({"a": 8, "b": 3}, {"a": 3, "b": 8}, {"a": 12, "b": 12}):
        assert m.eval(env) == min(env["a"], env["b"])
        assert c.eval(env) == -(-env["a"] // env["b"])
        assert p.eval(env) == env["a"] * 3 * env["b"]
        assert s.eval(env) == (env["a"] if env["a"] > env["b"] else 7)
    assert (
        REQUIRED[0][0].eval({"a": 8}) is True and REQUIRED[0][0].eval({"a": 6}) is False
    )
    try:
        bool(a)
        assert False
    except TypeError:
        pass
    # Staged ceildiv never forms an intermediate beyond its inputs, so an i32
    # numerator near INT32_MAX cannot wrap.
    env = {"a": 2**31 - 1, "b": 2}
    assert c.eval(env) == 2**30

    def intermediates(v):
        if v.op in ("var", "const"):
            return [v.eval(env)]
        return [v.eval(env)] + [x for s in v.args for x in intermediates(s)]

    assert max(intermediates(c)) <= 2**31 - 1


# CHECK-LABEL: whole_array_tilers_stage
@construct_test
def whole_array_tilers_stage():
    """Build the GEMM's three tilings on symbolic M, K, N and a symbolic step."""
    m, k, n, n_aie_rows, n_aie_cols, tb_n_rows = 32, 32, 32, 4, 2, 2
    M, K, N, step = (Sym.var(x) for x in ("M", "K", "N", "step"))
    REQUIRED.clear()
    rep = N // n // n_aie_cols
    grids = {
        "A": TensorAccessPattern.full((M, K)).tile((m * 2, k)).group((1, K // k)).repeat(rep),
        "B": TensorAccessPattern.full((K, N))
        .tile((k, n))
        .group((K // k, rep), steps=(1, n_aie_cols), col_major=True),
        "C": TensorAccessPattern.full((M, N))
        .tile((m * n_aie_rows, n))
        .group((tb_n_rows, rep), steps=(1, n_aie_cols)),
    }
    tiles = {name: g[step] for name, g in grids.items()}
    counts = {name: g.num_steps for name, g in grids.items()}
    for name in tiles:
        assert is_sym(tiles[name].offset) and is_sym(counts[name])
        try:
            len(grids[name])
            assert False
        except TypeError:
            pass
    n_guards = len(REQUIRED)
    assert n_guards > 0
    checked = 0
    for Mv, Kv, Nv in itertools.product((256, 512), (128, 256), (128, 256)):
        env = {"M": Mv, "K": Kv, "N": Nv}
        concrete = {
            "A": TensorAccessPattern.full((Mv, Kv))
            .tile((m * 2, k))
            .group((1, Kv // k))
            .repeat(Nv // n // n_aie_cols),
            "B": TensorAccessPattern.full((Kv, Nv))
            .tile((k, n))
            .group(
                (Kv // k, Nv // n // n_aie_cols), steps=(1, n_aie_cols), col_major=True
            ),
            "C": TensorAccessPattern.full((Mv, Nv))
            .tile((m * n_aie_rows, n))
            .group((tb_n_rows, Nv // n // n_aie_cols), steps=(1, n_aie_cols)),
        }
        for name in tiles:
            assert counts[name].eval(env) == len(concrete[name])
            for s in range(len(concrete[name])):
                env["step"] = s
                got = evaluated_tap(tiles[name], env)
                want = concrete[name][s]
                # A unit repeat stays as a dimension on the staged path (rank is
                # structural), so compare the walks, and the numbers where the
                # ranks agree.
                assert got.compare_access_orders(want), (name, s, got, want)
                if len(got.sizes) == len(want.sizes):
                    assert got == want, (name, s, got, want)
                checked += 1
        env["step"] = 0
        check_requires(env, expect_ok=True)
    # A shape the tiler must refuse fails a guard.
    check_requires({"M": 100, "K": 128, "N": 128, "step": 0}, expect_ok=False)
    print(f"staged GEMM tiles checked={checked} guards={n_guards}")
    # CHECK: staged GEMM tiles checked={{[1-9][0-9]*}} guards={{[1-9][0-9]*}}


# CHECK-LABEL: partial_and_slices_stage
@construct_test
def partial_and_slices_stage():
    N, step, lo, hi = (Sym.var(x) for x in ("N", "step", "lo", "hi"))
    REQUIRED.clear()
    g = TensorAccessPattern.full((3, N)).tile((3, 2)).group((1, 7), steps=(1, 3), partial=True)
    t = g[step]
    assert t.sizes[0].op == "select"  # min(R, ceildiv(remaining, S)) as a select tree
    for Nv in (28, 40, 64):
        gc = TensorAccessPattern.full((3, Nv)).tile((3, 2)).group((1, 7), steps=(1, 3), partial=True)
        assert g.num_steps.eval({"N": Nv}) == len(gc)
        for s in range(len(gc)):
            got = evaluated_tap(t, {"N": Nv, "step": s})
            want = gc[s]
            # The staged path cannot know a per-step repeat resolved to 1, so
            # it keeps that dimension; the walks are identical either way.
            assert got.compare_access_orders(want), (Nv, s, got, want)
            if len(got.sizes) == len(want.sizes):
                assert got == want, (Nv, s, got, want)
    # Slicing with staged bounds.
    v = TensorAccessPattern.full((8, N))[2:6, lo:hi]
    for Nv, lov, hiv in ((16, 0, 16), (32, 4, 12), (64, 1, 63)):
        got = evaluated_tap(v, {"N": Nv, "lo": lov, "hi": hiv})
        assert got == TensorAccessPattern.full((8, Nv))[2:6, lov:hiv]
    # partition on a staged length.
    parts = TensorAccessPattern.full((1, N)).partition(4)
    p = parts[step]
    for Nv in (64, 4096):
        for s in range(4):
            got = evaluated_tap(p, {"N": Nv, "step": s})
            assert got == TensorAccessPattern.full((1, Nv)).partition(4)[s]
    check_requires({"N": 64, "step": 1, "lo": 4, "hi": 12}, expect_ok=True)


# CHECK-LABEL: shim_form_stage
@construct_test
def shim_form_stage():
    """_dma_form() keeps a staged walk's rank and pads to the shim form; validators become guards."""
    M, K = Sym.var("M"), Sym.var("K")
    REQUIRED.clear()
    t = TensorAccessPattern.full((M, K)).tile((32, 32))[Sym.var("step")]
    assert isinstance(t, TensorAccessPattern)
    d = t._dma_form()
    assert len(d.sizes) == 4 and d.sizes[:2] == [1, 1] and d.strides[:2] == [0, 0]
    env = {"M": 64, "K": 128, "step": 3}
    assert (
        evaluated_tap(t, env)
        == TensorAccessPattern.full((64, 128)).tile((32, 32))[3]
    )
    # The TensorAccessPattern validators recorded guards rather than branching.
    assert any(
        "sizes" in msg or "Offset" in msg or "divisible" in msg for _, msg in REQUIRED
    )
    check_requires(env, expect_ok=True)
    # A repeat keeps slot 0 on the staged path too.
    r = TensorAccessPattern.full((1, K)).repeat(3)._dma_form()
    assert r.sizes[0] == 3 and r.strides[0] == 0
    assert np.array_equal(
        evaluated_tap(r, {"K": 16}).access_order(),
        TensorAccessPattern.full((1, 16)).repeat(3).access_order(),
    )
    # Literal unit dimensions are dropped to fit, as on the concrete path.
    N = Sym.var("N")
    u = TensorAccessPattern((N,), Sym.var("off"), [1, 1, 1, 1, N], [0, 0, 0, 0, 1])
    u = u._dma_form()
    assert len(u.sizes) == 4 and u.sizes[:3] == [1, 1, 1]
    assert (
        evaluated_tap(u, {"N": 16, "off": 0})
        == TensorAccessPattern((16,), 0, [1, 1, 1, 1, 16], [0, 0, 0, 0, 1])
    )
    try:
        TensorAccessPattern(
            (N,), 0, [N, N, 1, N, N, N], [0, 0, 0, 0, 0, 1]
        )._dma_form()
        assert False
    except ValueError as e:
        assert "does not fit" in str(e)
