# test_jit_many_args.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

# End-to-end @iron.jit test for a design with more than 5 host buffers. The NPU
# firmware only pre-translates the first 5 host buffer addresses into the AIE
# address space; buffers beyond that have the translation offset folded into
# their DDR address patch by the compiler. This exercises that path through the
# top-level IRON API (@iron.jit + Runtime), where a user would hit it.
#
# The design is three independent workers, each summing two inputs into one
# output: 6 inputs + 3 outputs = 9 host buffers (> 5). Spreading the work across
# three workers keeps each tile within its DMA channel budget.

import pytest
import numpy as np
import aie.iron as iron

from aie.utils import tensor
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_

NUM_LANES = 3  # 3 lanes x (2 in + 1 out) = 9 host buffers (> 5)
N = 1024


# Sum two input tiles element-wise into one output tile.
def add_two(of_a, of_b, of_out, n):
    elem_a = of_a.acquire(1)
    elem_b = of_b.acquire(1)
    elem_out = of_out.acquire(1)
    for i in range_(n):
        elem_out[i] = elem_a[i] + elem_b[i]
    of_a.release(1)
    of_b.release(1)
    of_out.release(1)


@iron.jit
def design(
    a0: iron.In,
    b0: iron.In,
    a1: iron.In,
    b1: iron.In,
    a2: iron.In,
    b2: iron.In,
    c0: iron.Out,
    c1: iron.Out,
    c2: iron.Out,
):
    tile_ty = np.ndarray[(N,), np.dtype[np.int32]]

    of_a = [ObjectFifo(tile_ty, depth=2, name=f"a{i}") for i in range(NUM_LANES)]
    of_b = [ObjectFifo(tile_ty, depth=2, name=f"b{i}") for i in range(NUM_LANES)]
    of_c = [ObjectFifo(tile_ty, depth=2, name=f"c{i}") for i in range(NUM_LANES)]

    workers = [
        Worker(
            add_two,
            fn_args=[of_a[i].cons(), of_b[i].cons(), of_c[i].prod(), N],
        )
        for i in range(NUM_LANES)
    ]

    # fn_args order: all producer/consumer fifo handles, appended after the
    # sequence inputs. The body receives seq inputs first, then the fifo handles.
    fifo_args = (
        [of_a[i].prod() for i in range(NUM_LANES)]
        + [of_b[i].prod() for i in range(NUM_LANES)]
        + [of_c[i].cons() for i in range(NUM_LANES)]
    )

    def sequence(*args):
        # seq order matches the design signature: a0,b0,a1,b1,a2,b2,c0,c1,c2.
        seq_args = args[: 3 * NUM_LANES]
        a_prods = args[3 * NUM_LANES : 4 * NUM_LANES]
        b_prods = args[4 * NUM_LANES : 5 * NUM_LANES]
        c_conses = args[5 * NUM_LANES : 6 * NUM_LANES]
        for i in range(NUM_LANES):
            a_prods[i].fill(seq_args[2 * i])
            b_prods[i].fill(seq_args[2 * i + 1])
        for i in range(NUM_LANES):
            c_conses[i].drain(seq_args[2 * NUM_LANES + i], wait=True)

    rt = Runtime(
        sequence,
        [tile_ty] * (3 * NUM_LANES),
        fn_args=fifo_args,
    )
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def test_jit_many_args():
    a = [
        tensor(np.arange(N, dtype=np.int32) * (i + 1), dtype=np.int32)
        for i in range(NUM_LANES)
    ]
    b = [
        tensor(np.arange(N, dtype=np.int32) * (i + 10), dtype=np.int32)
        for i in range(NUM_LANES)
    ]
    c = [tensor(np.zeros(N, dtype=np.int32), dtype=np.int32) for _ in range(NUM_LANES)]

    design(a[0], b[0], a[1], b[1], a[2], b[2], c[0], c[1], c[2])

    for i in range(NUM_LANES):
        c[i].to("cpu")
        ref = np.arange(N, dtype=np.int32) * (i + 1) + np.arange(N, dtype=np.int32) * (
            i + 10
        )
        assert np.array_equal(c[i].numpy(), ref), f"lane {i} mismatch"
