# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Drive the staged branches of the TensorAccessPattern algebra on real MLIR values.

A function's i32 arguments stand in for the runtime scalars of a dynamic
runtime sequence; the printed IR shows what the algebra emits. To evaluate a
staged pattern, the same builder runs on i32 constants and canonicalize folds
every offset, size and stride to a constant, which must equal the pattern
built on Python ints. A guard that holds folds away; one that fails is left
behind as a `cf.assert`.
"""

import itertools

from aie.dialects import func
from aie.dialects.aie import AIEDevice, device, object_fifo, tile
from aie.dialects.aiex import runtime_sequence, shim_dma_single_bd_task
from aie.extras import types as T
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects.arith import ScalarValue, constant
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import require
from aie.ir import Context, IntegerAttr, InsertionPoint, Location, Module, Value
from aie.passmanager import PassManager
from util import construct_test

# RUN: %python %s | FileCheck %s


def fields(tap):
    return [*tap.tensor_dims, tap.offset, *tap.sizes, *tap.strides]


def staged(build, names, values=None):
    """Build a function returning the fields of the patterns `build(**scalars)` makes.

    The scalars are i32 arguments, or i32 constants of `values` when given.
    Returns the module and the patterns.
    """
    module = Module.create()
    taps = []
    with InsertionPoint(module.body):
        arg_types = [] if values is not None else [T.i32()] * len(names)

        @func.FuncOp.from_py_func(*arg_types, name="staged")
        def _(*args):
            if values is not None:
                scalars = [constant(v, T.i32()) for v in values]
            else:
                scalars = [ScalarValue(a) for a in args]
            taps.extend(build(**dict(zip(names, scalars))))
            return [
                x if isinstance(x, Value) else constant(x, T.i32())
                for t in taps
                for x in fields(t)
            ]

    return module, taps


def evaluate(build, **values):
    """Return the concrete patterns `build` makes on `values`, and its failing guards."""
    module, taps = staged(build, list(values), list(values.values()))
    PassManager.parse("builtin.module(func.func(canonicalize))").run(module.operation)
    body = module.body.operations[0].regions[0].blocks[0]
    ops = list(body.operations)
    for v in ops[-1].operands:
        assert v.owner.name == "arith.constant", v.owner
    ints = iter(
        IntegerAttr(v.owner.attributes["value"]).value for v in ops[-1].operands
    )
    concrete = []
    for t in taps:
        dims = [next(ints) for _ in t.tensor_dims]
        offset = next(ints)
        sizes = [next(ints) for _ in t.sizes]
        strides = [next(ints) for _ in t.strides]
        concrete.append(TensorAccessPattern(dims, offset, sizes, strides))
    failed = [str(op.attributes["msg"]) for op in ops if op.name == "cf.assert"]
    return concrete, failed


# CHECK-LABEL: guards_stage
@construct_test
def guards_stage():
    def build(a, lo, hi):
        require(a % 4 == 0, "a must be a multiple of 4")
        return [TensorAccessPattern.full((8, a))[2:6, lo:hi]]

    with Context(), Location.unknown():
        module, (tap,) = staged(build, ["a", "lo", "hi"])
        assert tap.is_symbolic and tap.sizes[0] == 4
        print(module)
        _, failed = evaluate(build, a=16, lo=4, hi=12)
        assert failed == []
        _, failed = evaluate(build, a=18, lo=4, hi=20)
        print(failed)
    # CHECK: func.func @staged(%[[A:.*]]: i32, %[[LO:.*]]: i32, %[[HI:.*]]: i32)
    # CHECK: arith.remsi %[[A]]
    # CHECK: cf.assert %{{.*}}, "a must be a multiple of 4"
    # CHECK: cf.assert %{{.*}}, "slice start must be >= 0 on a runtime dimension"
    # CHECK: cf.assert %{{.*}}, "slice stop exceeds the dimension"
    # CHECK: cf.assert %{{.*}}, "slice selects no elements"
    # CHECK: ['"a must be a multiple of 4"', '"slice stop exceeds the dimension"']


# CHECK-LABEL: loop_index_stage
@construct_test
def loop_index_stage():
    """A `range_` induction variable is an index; it indexes a pattern as an i64."""
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):

            @func.FuncOp.from_py_func(T.index(), name="loop")
            def _(iv):
                chunk = TensorAccessPattern.full((1, 4096)).partition(8)[iv]
                return [chunk.offset]

        print(module)
    # CHECK: func.func @loop(%[[IV:.*]]: index) -> i64
    # CHECK: arith.index_cast %[[IV]] : index to i64


# CHECK-LABEL: whole_array_tilings_fold
@construct_test
def whole_array_tilings_fold():
    """The GEMM's tilings on staged M, K, N and a staged row block."""
    m, k, n, n_aie_rows, n_aie_cols, rows = 32, 32, 32, 4, 2, 2

    def build(M, K, N, step):
        require(M % (m * n_aie_rows) == 0, "M must be a multiple of m * n_aie_rows")
        require(K % k == 0, "K must be a multiple of k")
        require(N % (n * n_aie_cols) == 0, "N must be a multiple of n * n_aie_cols")
        A_tiles = TensorAccessPattern.full((M, K)).tile((m * 2, k))
        B_tiles = TensorAccessPattern.full((K, N)).tile((k, n)).permute((1, 0, 2, 3))
        C_tiles = TensorAccessPattern.full((M, N)).tile((m * n_aie_rows, n))
        return [A_tiles[step * 2 + 1].repeat(N // n // n_aie_cols)] + [
            t
            for col in range(n_aie_cols)
            for t in (
                B_tiles[col::n_aie_cols],
                C_tiles[step * rows : step * rows + rows, col::n_aie_cols],
            )
        ]

    checked = 0
    with Context(), Location.unknown():
        for Mv, Kv, Nv in itertools.product((256, 512), (128, 256), (128, 256)):
            for s in range(Mv // (m * n_aie_rows) // rows):
                got, failed = evaluate(build, M=Mv, K=Kv, N=Nv, step=s)
                assert failed == [], failed
                assert got == build(Mv, Kv, Nv, s), (Mv, Kv, Nv, s)
                checked += 1
        _, failed = evaluate(build, M=100, K=128, N=128, step=0)
    print(f"staged GEMM steps checked={checked}")
    print(failed[0])
    # CHECK: staged GEMM steps checked=12
    # CHECK: "M must be a multiple of m * n_aie_rows"


# CHECK-LABEL: slices_and_partition_fold
@construct_test
def slices_and_partition_fold():
    def build(N, step, lo, hi):
        tiles = TensorAccessPattern.full((3, N)).tile((3, 2))
        return [
            *(tiles[0, j::3] for j in range(3)),
            TensorAccessPattern.full((8, N))[2:6, lo:hi],
            TensorAccessPattern.full((1, N)).partition(4)[step],
        ]

    with Context(), Location.unknown():
        for Nv, lo, hi in ((28, 0, 16), (40, 4, 12), (64, 1, 63)):
            for s in range(4):
                got, failed = evaluate(build, N=Nv, step=s, lo=lo, hi=hi)
                assert failed == [], failed
                assert got == build(Nv, s, lo, hi), (Nv, s)
    print("ragged slices and partitions fold to the concrete patterns")
    # CHECK: ragged slices and partitions fold to the concrete patterns


# CHECK-LABEL: shim_form_stage
@construct_test
def shim_form_stage():
    """A staged walk keeps its rank in a shim BD, padded to four dimensions."""
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu2)
        def _():
            of = object_fifo("of", tile(0, 0), tile(0, 2), 2, T.memref(32, T.i32()))

            @runtime_sequence(T.memref(4096, T.i32()), T.i32(), T.i32(), T.i32())
            def _(buf, K, step, N):
                tile_tap = TensorAccessPattern.full((64, K)).tile((32, 32))[0, step]
                shim_dma_single_bd_task(of, buf, tap=tile_tap)
                repeated = TensorAccessPattern.full((1, K)).repeat(3)
                shim_dma_single_bd_task(of, buf, tap=repeated)
                too_deep = TensorAccessPattern(
                    (N,), 0, [N, N, 1, N, N, N], [0, 0, 0, 0, 0, 1]
                )
                try:
                    shim_dma_single_bd_task(of, buf, tap=too_deep)
                    assert False
                except ValueError as e:
                    print(e)

        print(ctx.module)
    # CHECK: a DMA BD with more than 4 dimensions (got 5) needs constant sizes and strides
    # CHECK: runtime_sequence
    # CHECK: aie.dma_bd({{.*}} sizes = [1, 1, 32, 32] strides = [0, 0, %{{.*}}, 1])
    # CHECK: aie.dma_bd({{.*}} sizes = [3, 1, 1, %{{.*}}] strides = [0, 0, 0, 1])
    # CHECK: repeat_count = 2
