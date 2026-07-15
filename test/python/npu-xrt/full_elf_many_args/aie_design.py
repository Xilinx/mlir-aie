# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Full-ELF design with MORE THAN 5 host buffers, used to demonstrate the DDR
# address-patch aperture-offset bug (see companion test.py).
#
# N_BUFFERS independent workers each write a distinct constant (100 + i) into
# its own output ObjectFifo; the runtime sequence drains each FIFO into its own
# host buffer. This produces N_BUFFERS runtime-sequence arguments (arg_idx
# 0..N-1), so it exercises host arguments beyond the first 5.
#
# Since all .py files are picked up as a test, add this to not execute this
# design file directly.
# REQUIRES: dont_run
# RUN: echo

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2Col4
from aie.dialects.aiex import npu_load_pdi

N_BUFFERS = 8
out_ty = np.ndarray[(1,), np.dtype[np.int32]]


def design():
    device_name = "main"
    fifos = [ObjectFifo(out_ty, name=f"of_out{i}") for i in range(N_BUFFERS)]

    def make_core_fn(val):
        def core_fn(of_out):
            out_elem = of_out.acquire(1)
            out_elem[0] = val
            of_out.release(1)

        return core_fn

    workers = [
        Worker(make_core_fn(100 + i), [fifos[i].prod()], while_true=False)
        for i in range(N_BUFFERS)
    ]

    rt = Runtime()
    with rt.sequence(*([out_ty] * N_BUFFERS)) as tensors:
        rt.inline_ops(lambda: npu_load_pdi(device_ref=device_name), [])
        for w in workers:
            rt.start(w)
        for i in range(N_BUFFERS):
            rt.drain(fifos[i].cons(), tensors[i], wait=True)

    return str(Program(NPU2Col4(), rt).resolve_program(device_name=device_name))


print(design())
