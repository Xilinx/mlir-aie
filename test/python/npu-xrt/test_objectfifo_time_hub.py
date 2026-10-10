# test_objectfifo_time_hub.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""Two pipeline stages time-share one hub core through a merge and a dispatch.

Each token goes stage 1 -> hub -> stage 2 -> hub -> host. The hub's input is a
merge of both stages' outputs on one mem tile channel, and its output is a
dispatch on one mem tile channel whose turns alternate between stage 2 and the
host. The runtime issues a token only once the previous one is back, so the
hub sees stage 1 and stage 2 in turn, which is what lets the dispatch's fixed
order send each result to the right place.
"""

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_

TOKENS = 4
N = 16


@iron.jit
def time_hub(x: In, y: Out):
    obj = np.ndarray[(N,), np.dtype[np.int32]]

    of_x = ObjectFifo(obj, depth=1, name="x")
    hub_in = ObjectFifo(obj, depth=2, name="hub_in")
    from_s1, from_s2 = hub_in.prod().merge(2, names=["from_s1", "from_s2"])
    hub_out = ObjectFifo(obj, depth=2, name="hub_out")
    to_s2, to_host = hub_out.cons().dispatch(2, names=["to_s2", "to_host"])

    def stage(of_in, of_out, k):
        for _ in range_(TOKENS):
            a = of_in.acquire(1)
            b = of_out.acquire(1)
            for i in range_(N):
                b[i] = a[i] + k
            of_in.release(1)
            of_out.release(1)

    def hub(of_in, of_out):
        for _ in range_(2 * TOKENS):
            a = of_in.acquire(1)
            b = of_out.acquire(1)
            for i in range_(N):
                b[i] = a[i] * 2
            of_in.release(1)
            of_out.release(1)

    workers = [
        Worker(stage, fn_args=[of_x.cons(), from_s1.prod(), 1]),
        Worker(hub, fn_args=[hub_in.cons(), hub_out.prod()]),
        Worker(stage, fn_args=[to_s2.cons(), from_s2.prod(), 3]),
    ]

    tensor = np.ndarray[(TOKENS * N,), np.dtype[np.int32]]

    def seq(x, y, x_h, y_h):
        for t in range(TOKENS):
            token = TensorAccessPattern(
                (TOKENS * N,), offset=t * N, sizes=[1, 1, 1, N], strides=[0, 0, 0, 1]
            )
            x_h.fill(x, tap=token)
            y_h.drain(y, tap=token, wait=True)

    rt = Runtime(seq, [tensor, tensor, of_x.prod(), to_host.cons()])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def test_objectfifo_time_hub():
    x = iron.arange(TOKENS * N, dtype=np.int32)
    y = iron.zeros(TOKENS * N, dtype=np.int32, device="npu")
    time_hub(x, y)
    y.to("cpu")

    expected = ((np.arange(TOKENS * N, dtype=np.int32) + 1) * 2 + 3) * 2
    np.testing.assert_array_equal(y.numpy(), expected)
