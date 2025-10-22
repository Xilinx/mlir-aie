# passthrough_kernel/passthrough_kernel.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2025 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2, Tile

def my_passthrough_kernel(dev, in1_size, out_size, ncores=1):
    assert ncores == 1 or ncores == 2
    assert in1_size * ncores == out_size

    in1_dtype = np.uint8
    out_dtype = np.uint8

    # Define tensor types
    line_size = in1_size // in1_dtype(0).nbytes
    line_out_size = out_size // out_dtype(0).nbytes
    line_type = np.ndarray[(line_size,), np.dtype[in1_dtype]]
    line_out_type = np.ndarray[(line_out_size,), np.dtype[out_dtype]]

    # Dataflow with ObjectFifos
    of_in, of_out = [], []
    for i in range(ncores):
        of_in.append(ObjectFifo(line_type, name=f"in{i}"))
    of_out_shim = ObjectFifo(line_out_type, name=f"out{i}")
    of_out = of_out_shim.prod().join(
        offsets=[line_size * i for i in range(ncores)], obj_types=[line_type]*ncores
    )

    # External, binary kernel definition
    passthrough_fn = Kernel(
        "passThroughLine",
        "passThrough.cc.o",
        [line_type, line_type, np.int32],
    )

    # Task for the core to perform
    def core_fn(of_in, of_out, passThroughLine):
        elemOut = of_out.acquire(1)
        elemIn = of_in.acquire(1)
        passThroughLine(elemIn, elemOut, line_size)
        of_in.release(1)
        of_out.release(1)

    # Create a worker to perform the task
    workers = []
    for i in range(ncores):
        # TODO: placer fails unless I put placement here... why??
        my_worker = Worker(core_fn, [of_in[i].cons(), of_out[i].prod(), passthrough_fn], placement=Tile(0, 2+i))
        workers.append(my_worker)

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    rt_types = [line_type] * ncores + [line_out_type]

    with rt.sequence(*rt_types) as in_outs:
        rt.start(*workers)
        for i in range(ncores):
            rt.fill(of_in[i].prod(), in_outs[i])
        rt.drain(of_out_shim.cons(), in_outs[-1], wait=True)

    # Place components (assign the resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
p.add_argument(
    "-i1s", "--in1_size", required=True, dest="in1_size", type=int, help="Input 1 size"
)
p.add_argument("-os", "--out_size", required=True, dest="out_size", type=int, help="Output size")
p.add_argument("--ncores", type=int, required=True, help="1 or 2")
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = NPU1Col1()
elif opts.device == "npu2":
    dev = NPU2()
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(opts.device))

if opts.in1_size % 64 != 0 or opts.in1_size < 512:
    print(f"In1 buffer size ({in1_size}) must be a multiple of 64 and greater than or equal to 512")
    raise ValueError
print(my_passthrough_kernel(dev, opts.in1_size, opts.out_size, opts.ncores))
