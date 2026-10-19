# test_operator_resolution.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""On-device test of an Operator resolved for the NPU it runs on.

``Scale`` declares its size and factor, a tile it may be given, and a lane
count only the device decides. Resolved for this NPU, its operands report the
streams a design builds, and the design built from the resolved values
scales every element.
"""

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.utils.compile.jit.operator import Operator
from aie.utils.compile.jit.params import param, tunable


class Scale(Operator):
    n: int = param()
    factor: int = param(default=3)
    tile: int | None = tunable()
    lanes: int | None = tunable(init=False)
    x = In(n, dtype=np.int32, obj_shape=tile, per=lanes)
    y = Out(n, dtype=np.int32, obj_shape=tile, per=lanes)

    def resolve(self, dev):
        return self.replace(tile=self.tile or 64, lanes=min(dev.cols, 4))


@iron.jit
def scale(
    a: In,
    c: Out,
    *,
    n: CompileTime[int],
    factor: CompileTime[int],
    tile: CompileTime[int],
    lanes: CompileTime[int],
):
    tile_ty = np.ndarray[(tile,), np.dtype[np.int32]]
    tensor_ty = np.ndarray[(n,), np.dtype[np.int32]]
    share = n // lanes
    ins = [ObjectFifo(tile_ty, name=f"in{lane}") for lane in range(lanes)]
    outs = [ObjectFifo(tile_ty, name=f"out{lane}") for lane in range(lanes)]

    def core(cin, cout):
        for _ in range_(share // tile):
            x = cin.acquire(1)
            y = cout.acquire(1)
            for i in range_(tile):
                y[i] = x[i] * factor
            cin.release(1)
            cout.release(1)

    def seq(A, C, *ends):
        shares = TensorAccessPattern.full((n,)).split(0, share)
        for lane in range(lanes):
            ends[lane].fill(A, tap=shares[lane])
        for lane in range(lanes):
            ends[lanes + lane].drain(C, tap=shares[lane], wait=True)

    workers = [Worker(core, [i.cons(), o.prod()]) for i, o in zip(ins, outs)]
    rt = Runtime(
        seq, [tensor_ty, tensor_ty, *(i.prod() for i in ins), *(o.cons() for o in outs)]
    )
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def test_an_operator_resolved_for_this_npu_builds_a_design_that_runs():
    dev = iron.get_current_device()
    op = Scale(n=1024).resolved(dev)
    assert (op.tile, op.lanes) == (64, min(dev.cols, 4))
    assert op.x.shape == (1024,) and op.x.obj_shape == (64,)
    assert op.x.lanes == op.y.lanes == op.lanes
    assert op.resolved(type(dev)()) is op
    assert op.design_key()[1][0] == ("device", dev.name)

    host = np.random.default_rng(6).integers(-(2**20), 2**20, op.n, dtype=np.int32)
    a = iron.tensor(host, dtype=np.int32, device="npu")
    c = iron.zeros(op.n, dtype=np.int32, device="npu")
    scale(a, c, n=op.n, factor=op.factor, tile=op.tile, lanes=op.lanes)
    c.to("cpu")
    np.testing.assert_array_equal(c.numpy(), host * op.factor)
