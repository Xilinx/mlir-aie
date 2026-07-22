# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# Pins down the IRON-side contract for `Buffer(address=)`: the requested L1
# address must survive resolution and appear as the `address` attribute on the
# emitted `aie.buffer`. This is the surface a host-driven design uses to place a
# runtime-written buffer at a fixed address an external runtime pokes, instead
# of leaving it to compiler assignment.

import numpy as np
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker, Buffer

from aie.iron.device import NPU2Col1


# CHECK:  module {
# CHECK:    aie.device(npu2_1col) {
# CHECK:      %pinned_local_buf = aie.buffer(%{{.*}}) {address = 8192 : i32, sym_name = "pinned_local_buf"} : memref<4096xui8>
# CHECK:      %default_local_buf = aie.buffer(%{{.*}}) {sym_name = "default_local_buf"} : memref<4096xui8>
def passthrough_pinned_buff():
    in1_size = 4096
    in1_dtype = np.uint8

    # Define tensor types
    line_size = in1_size // in1_dtype(0).nbytes
    line_type = np.ndarray[(line_size,), np.dtype[in1_dtype]]
    vector_type = np.ndarray[(line_size,), np.dtype[in1_dtype]]

    # Dataflow with ObjectFifos
    of_in = ObjectFifo(line_type, name="in")
    of_out = ObjectFifo(line_type, name="out")

    # External, binary kernel definition
    passthrough_fn = Kernel(
        "passThroughLine",
        "passThrough.cc.o",
        [line_type, line_type, np.int32],
    )

    # This buffer is pinned to a fixed L1 address.
    pinned_buf = Buffer(line_type, name="pinned_local_buf", address=0x2000)
    # This buffer is left to compiler assignment (no address attribute emitted).
    default_buf = Buffer(line_type, name="default_local_buf")

    # Task for the core to perform
    def core_fn(of_in, of_out, buf1, buf2, passThroughLine):
        elemOut = of_out.acquire(1)
        elemIn = of_in.acquire(1)
        passThroughLine(elemIn, buf1, line_size)
        passThroughLine(buf1, buf2, line_size)
        passThroughLine(buf2, elemOut, line_size)
        of_in.release(1)
        of_out.release(1)

    # Create a worker to perform the task
    my_worker = Worker(
        core_fn,
        [of_in.cons(), of_out.prod(), pinned_buf, default_buf, passthrough_fn],
    )

    # Runtime operations to move data to/from the AIE-array
    def sequence(a_in, b_out, _, in_h, out_h):
        in_h.fill(a_in)
        out_h.drain(b_out, wait=True)

    rt = Runtime(
        sequence,
        [vector_type, vector_type, vector_type],
        fn_args=[of_in.prod(), of_out.cons()],
    )

    # Place components (assign them resources on the device) and generate an MLIR module
    return Program(NPU2Col1(), rt, workers=[my_worker]).resolve_program()


print(passthrough_pinned_buff())
