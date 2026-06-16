# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
#
#
# Minimal full-ELF flow example using unplaced IRON.
#
# The core writes a single i32 value (42) to an output ObjectFIFO. The runtime
# sequence drains the FIFO into a 4-byte host buffer via a single DMA task.
#
# Usage:
#   python3 aie_design.py > aie.mlir
#
# Since all .py files are picked up as a test, add this to not execute this design file
# REQUIRES: dont_run
# RUN: echo

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2Col1
from aie.dialects.aiex import npu_load_pdi


def design():
    device_name = "main"

    out_ty = np.ndarray[(1,), np.dtype[np.int32]]

    of_out = ObjectFifo(out_ty, name="objfifo_out")

    def core_fn(of_out):
        out_elem = of_out.acquire(1)
        out_elem[0] = 42
        of_out.release(1)

    worker = Worker(core_fn, [of_out.prod()], while_true=False)

    rt = Runtime()

    def sequence(out_tensor):
        rt.inline_ops(lambda: npu_load_pdi(device_ref=device_name), [])
        of_out.cons().drain(out_tensor, wait=True)

    rt.sequence(sequence, [out_ty])

    mlir = Program(NPU2Col1(), rt, workers=[worker]).resolve_program(
        device_name=device_name
    )
    return str(mlir)


print(design())
