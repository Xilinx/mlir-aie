# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s
import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker, LocalBuffer
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2Col1


# CHECK:  module {
# CHECK:    aie.device(npu2_1col) {
# CHECK:      %tile_0_2 = aie.tile(0, 2)
# CHECK:      %unint_local_buf = aie.buffer(%tile_0_2) {sym_name = "unint_local_buf"} : memref<4096xui8>
# CHECK:      %init_local_buf = aie.buffer(%tile_0_2) {sym_name = "init_local_buf"} : memref<4096xui8> = dense<0>
def passthrough_local_buff():
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

    # Task for the core to perform
    def core_fn(of_in, of_out, passThroughLine):
        # This buffer is local to the core and has no initial value
        unint_local_buf = LocalBuffer(line_type, name="unint_local_buf")
        # This buffer is local to the core and has an initial value
        init_local_buf = LocalBuffer(
            line_type,
            name="init_local_buf",
            initial_value=np.zeros(line_size, dtype=in1_dtype),
        )
        elemOut = of_out.acquire(1)
        elemIn = of_in.acquire(1)
        passThroughLine(elemIn, unint_local_buf, line_size)
        passThroughLine(unint_local_buf, init_local_buf, line_size)
        passThroughLine(init_local_buf, elemOut, line_size)
        of_in.release(1)
        of_out.release(1)

    # Create a worker to perform the task
    my_worker = Worker(
        core_fn,
        [of_in.cons(), of_out.prod(), passthrough_fn],
    )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(vector_type, vector_type) as (a_in, b_out):
        rt.start(my_worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), b_out, wait=True)

    # Place components (assign them resources on the device) and generate an MLIR module
    return Program(NPU2Col1(), rt).resolve_program(SequentialPlacer())


print(passthrough_local_buff())
