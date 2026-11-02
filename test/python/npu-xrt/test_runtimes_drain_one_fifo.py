# test_runtimes_drain_one_fifo.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""On-device test of several runtimes filling and draining one pair of fifos.

Each runtime of one configuration sends one record of the input through a
Worker that adds one, and drains the record back. A coordinator runs them in
turn. Every runtime takes its own ``cons()`` of the output fifo, which must
still leave the fifo one shim end: a second end receives the same stream with
no buffer descriptor and stalls the Worker.
"""

import aie.iron as iron
import numpy as np
from aie.iron import (
    DeviceConfiguration,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_

REC = 64
STEPS = 3


@iron.jit(full_elf=True)
def steps(a: In, b: Out):
    rec_ty = np.ndarray[(REC,), np.dtype[np.int32]]
    all_ty = np.ndarray[(STEPS * REC,), np.dtype[np.int32]]
    of_in = ObjectFifo(rec_ty, name="into")
    of_out = ObjectFifo(rec_ty, name="out_of")

    def core(c_in, c_out):
        x = c_in.acquire(1)
        y = c_out.acquire(1)
        for i in range_(REC):
            y[i] = x[i] + 1
        c_in.release(1)
        c_out.release(1)

    worker = Worker(core, [of_in.cons(), of_out.prod()])

    def step(a, b, into, out_of):
        into.fill(a)
        out_of.drain(b, wait=True)

    runtimes = [
        Runtime(step, [rec_ty, rec_ty, of_in.prod(), of_out.cons()], name=f"step{k}")
        for k in range(STEPS)
    ]

    def coordinate(A, B):
        with pack.configure():
            for k, runtime in enumerate(runtimes):
                runtime.call(A.window(k * REC, (REC,)), B.window(k * REC, (REC,)))

    coordinator = Runtime(
        coordinate, [all_ty, all_ty], name="coordinate", implicit_configure=False
    )
    pack = DeviceConfiguration(
        "pack",
        iron.get_current_device(),
        workers=[worker],
        runtimes=[*runtimes, coordinator],
    )
    return Program.compose([pack], entry=coordinator).resolve_program()


def test_runtimes_draining_one_fifo_each_get_their_record():
    data = np.random.default_rng(3).integers(-1000, 1000, STEPS * REC, dtype=np.int32)
    a = iron.tensor(data, dtype=np.int32, device="npu")
    b = iron.zeros(STEPS * REC, dtype=np.int32, device="npu")
    steps(a, b)
    b.to("cpu")
    np.testing.assert_array_equal(b.numpy(), data + 1)
