# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# The IRON-side contract for Buffer(mem_bank=): the requested L1 bank survives
# resolution and appears as the mem_bank attribute on the emitted aie.buffer,
# which the allocator treats as a hard constraint.

import numpy as np
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker, Buffer

from aie.iron.device import NPU2Col1


# CHECK:  module {
# CHECK:    aie.device(npu2_1col) {
# CHECK:      %pinned_bank_buf = aie.buffer(%{{.*}}) {mem_bank = 1 : i32, sym_name = "pinned_bank_buf"} : memref<4096xui8>
# CHECK:      %default_bank_buf = aie.buffer(%{{.*}}) {sym_name = "default_bank_buf"} : memref<4096xui8>
def passthrough_pinned_bank():
    in1_size = 4096
    in1_dtype = np.uint8

    line_size = in1_size // in1_dtype(0).nbytes
    line_type = np.ndarray[(line_size,), np.dtype[in1_dtype]]
    vector_type = np.ndarray[(line_size,), np.dtype[in1_dtype]]

    of_in = ObjectFifo(line_type, name="in")
    of_out = ObjectFifo(line_type, name="out")

    passthrough_fn = Kernel(
        "passThroughLine",
        "passThrough.cc.o",
        [line_type, line_type, np.int32],
    )

    # This buffer is pinned to a fixed L1 bank.
    pinned_buf = Buffer(line_type, name="pinned_bank_buf", mem_bank=1)
    # This buffer is left to compiler assignment (no mem_bank attribute emitted).
    default_buf = Buffer(line_type, name="default_bank_buf")

    def core_fn(of_in, of_out, buf1, buf2, passThroughLine):
        elemOut = of_out.acquire(1)
        elemIn = of_in.acquire(1)
        passThroughLine(elemIn, buf1, line_size)
        passThroughLine(buf1, buf2, line_size)
        passThroughLine(buf2, elemOut, line_size)
        of_in.release(1)
        of_out.release(1)

    my_worker = Worker(
        core_fn,
        [of_in.cons(), of_out.prod(), pinned_buf, default_buf, passthrough_fn],
    )

    def sequence(a_in, b_out, _, in_h, out_h):
        in_h.fill(a_in)
        out_h.drain(b_out, wait=True)

    rt = Runtime(
        sequence,
        [vector_type, vector_type, vector_type, of_in.prod(), of_out.cons()],
    )

    return Program(NPU2Col1(), rt, workers=[my_worker]).resolve_program()


print(passthrough_pinned_bank())


# Buffer validates mem_bank the way Worker validates stack_size and
# reserved_data_size: a ValueError at construction time, before the value
# reaches MLIR.
try:
    Buffer(np.ndarray[(4,), np.dtype[np.uint8]], mem_bank=-1)
    raise AssertionError("expected ValueError for mem_bank < 0")
except ValueError:
    pass

try:
    Buffer(np.ndarray[(4,), np.dtype[np.uint8]], mem_bank="1")
    raise AssertionError("expected ValueError for non-int mem_bank")
except ValueError:
    pass

try:
    Buffer(np.ndarray[(4,), np.dtype[np.uint8]], address=-1)
    raise AssertionError("expected ValueError for address < 0")
except ValueError:
    pass

try:
    Buffer(np.ndarray[(4,), np.dtype[np.uint8]], address="0x1000")
    raise AssertionError("expected ValueError for non-int address")
except ValueError:
    pass

# CHECK: PASS
print("PASS")
