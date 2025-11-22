#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

from ml_dtypes import bfloat16
import numpy as np
import argparse
import sys

from aie.extras.dialects.ext import arith
from aie.dialects.aie import T
from aie.iron import (
    Kernel,
    ObjectFifo,
    Program,
    GlobalBuffer,
    Runtime,
    Worker,
    WorkerRuntimeBarrier,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.controlflow import range_
from aie.helpers.util import np_ndarray_type_get_shape

#
# Scale Shift is a multi-core example that time-multiplexes the
# cores to perform first a scale, followed by a bias addition
# to shift: A * B + C = D, where each is a vector of bfloat16
# values. A "Runtime Parameter" rtp is used to switch between
#  * and + in each core at runtime. WorkerRuntimeBarriers are
# used to synchronize between the runtime sequence and each Worker.
#


def my_scale_shift(dev, in1_size, in2_size, in3_size, out_size, trace_size):
    in1_dtype = bfloat16
    in2_dtype = bfloat16
    in3_dtype = bfloat16
    out_dtype = bfloat16

    tensor_size = in1_size // in1_dtype(0).nbytes

    # Tile sizes
    tile_size = 1024
    tensor_div_tile = tensor_size // tile_size

    # This example 2 cores to paralelize the scale and
    # shift operations one after another.
    # The number of cores can be changed to 1 or 4.
    n_cores = 2
    tiles = tensor_div_tile // n_cores

    assert in2_size == in1_size, "input2 buffer size must match input1 buffer size."
    assert in3_size == in1_size, "input3 buffer size must match input1 buffer size."
    assert out_size == in1_size, "Output buffer size must match input1 buffer size."

    # The trace size is the size of the trace buffer.
    # The trace buffer is used to store the trace of the
    # operations performed by the AIE array.
    enable_trace = 1 if trace_size > 0 else 0

    # Type used in the external memory
    tensor_ty = np.ndarray[(tensor_size,), np.dtype[out_dtype]]
    # Type used in the tile memory
    tile_ty = np.ndarray[(tile_size,), np.dtype[out_dtype]]

    # Type used in the tile memory
    A_ty = np.ndarray[(tile_size,), np.dtype[in1_dtype]]
    B_ty = np.ndarray[(tile_size,), np.dtype[in2_dtype]]
    C_ty = np.ndarray[(tile_size,), np.dtype[out_dtype]]

    # Type used in the memory tile which aggregates across the 2 cores
    A_memTile_ty = np.ndarray[(tile_size * n_cores,), np.dtype[in1_dtype]]
    B_memTile_ty = np.ndarray[(tile_size * n_cores,), np.dtype[in2_dtype]]
    C_memTile_ty = np.ndarray[(tile_size * n_cores,), np.dtype[out_dtype]]

    # AIE Core Function declarations
    scale_shift_bf16 = Kernel(
        "eltwise_mul_add_bf16_vector",
        "scale_shift.o",
        [tile_ty, tile_ty, tile_ty, np.int32],
    )

    # AIE-array data movement with object fifos
    # Input A
    inA = ObjectFifo(A_memTile_ty, name="inA")
    of_offsets = [
        (np.prod(np_ndarray_type_get_shape(A_memTile_ty)) // n_cores) * i
        for i in range(n_cores)
    ]
    inA_fifos = inA.cons().split(
        of_offsets,
        obj_types=[A_ty] * n_cores,
        names=[f"memA{i}" for i in range(n_cores)],
    )

    # Input B
    inB = ObjectFifo(B_memTile_ty, name="inB")
    of_offsets = [
        (np.prod(np_ndarray_type_get_shape(B_memTile_ty)) // n_cores) * i
        for i in range(n_cores)
    ]
    inB_fifos = inB.cons().split(
        of_offsets,
        obj_types=[B_ty] * n_cores,
        names=[f"memB{i}" for i in range(n_cores)],
    )

    # Output C
    outC = ObjectFifo(C_memTile_ty, name="outC")
    of_offsets = [
        (np.prod(np_ndarray_type_get_shape(C_memTile_ty)) // n_cores) * i
        for i in range(n_cores)
    ]
    outC_fifos = outC.prod().join(
        of_offsets,
        obj_types=[C_ty] * n_cores,
        names=[f"memC{i}" for i in range(n_cores)],
    )

    # Runtime parameters
    rtps = []
    for i in range(n_cores):
        rtps.append(
            GlobalBuffer(
                np.ndarray[(1,), np.dtype[np.int32]],
                name=f"rtp{i}",
                initial_value=np.array([1], dtype=np.int32),
                use_write_rtp=True,
            )
        )

    # Create barriers to synchronize individual workers with the runtime sequence
    workerBarriers = []
    for i in range(n_cores):
        workerBarriers.append(WorkerRuntimeBarrier())

    # Task for the cores to perform
    def core_fn(of_a, of_b, of_c, mul_add, my_rtp, barrier):
        barrier.wait_for_value(1)
        is_mul = my_rtp[0]
        for _ in range_(tiles):
            elem_out = of_c.acquire(1)
            elem_in_a = of_a.acquire(1)
            elem_in_b = of_b.acquire(1)
            mul_add(elem_in_a, elem_in_b, elem_out, is_mul)
            of_a.release(1)
            of_b.release(1)
            of_c.release(1)
        barrier.release_with_value(1)

    # Set up workers to perform the tasks
    workers = []
    for i in range(n_cores):
        workers.append(
            Worker(
                core_fn,
                fn_args=[
                    inA_fifos[i].cons(),
                    inB_fifos[i].cons(),
                    outC_fifos[i].prod(),
                    scale_shift_bf16,
                    rtps[i],
                    workerBarriers[i],
                ],
                # trace=enable_trace,
            )
        )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty, tensor_ty) as (A, B, C, D):
        if enable_trace:
            rt.enable_trace(trace_size, workers=workers)

        rt.start(*workers)

        # Set runtime parameters
        # 1 == multiply
        def set_rtps(*args):
            for rtp in args:
                rtp[0] = 1

        rt.inline_ops(set_rtps, rtps)

        # Set the barriers to 1 to allow the worker to read the
        # runtime parameters and start the computation
        for i in range(n_cores):
            rt.set_barrier(workerBarriers[i], 1)

        # Fill the input objectFIFOs with data
        tg1 = rt.task_group()
        rt.fill(inA.prod(), A, task_group=tg1)
        rt.fill(inB.prod(), B, task_group=tg1)
        # Drain the output objectFIFOs to the external memory
        # Wait for the data to be drained before continuing
        rt.drain(outC.cons(), D, wait=True, task_group=tg1)
        rt.finish_task_group(tg1)

        # Set runtime parameters
        # 0 == add
        def set_rtps(*args):
            for rtp in args:
                rtp[0] = 0

        rt.inline_ops(set_rtps, rtps)

        # Set the barriers to 1 to allow the worker to read the
        # runtime parameters and start the computation
        for i in range(n_cores):
            rt.set_barrier(workerBarriers[i], 1)

        # Fill the input objectFIFOs with data
        # The input D is the output of the previous operation
        tg2 = rt.task_group()
        rt.fill(inA.prod(), D, task_group=tg2)
        rt.fill(inB.prod(), C, task_group=tg2)
        # Drain the output objectFIFOs to the external memory
        # Wait for the data to be drained before continuing
        rt.drain(outC.cons(), D, wait=True, task_group=tg2)
        rt.finish_task_group(tg2)

    # Place components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


p = argparse.ArgumentParser()
## Parse command line arguments

## The device name is used to select the device to use
## The device name can be either npu or npu2
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
## The input and output buffer sizes are used to set the size of the buffers
## The input and output buffer sizes:
##     * are in bytes
##     * must be a multiple of the tile size
##     * must have matching sizes
p.add_argument(
    "-i1s", "--in1_size", required=True, dest="in1_size", help="Input 1 size"
)
p.add_argument(
    "-i2s", "--in2_size", required=True, dest="in2_size", help="Input 2 size"
)
p.add_argument(
    "-i3s", "--in3_size", required=True, dest="in3_size", help="Input 3 size"
)
p.add_argument("-os", "--out_size", required=True, dest="out_size", help="Output size")
## The trace size is used to set the size of the trace buffer
p.add_argument(
    "-t",
    "--trace_size",
    required=False,
    dest="trace_size",
    default=0,
    help="Trace buffer size",
)
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = NPU1Col1()
elif opts.device == "npu2":
    dev = NPU2Col1()
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(opts.device))

in1_size = int(opts.in1_size)
in2_size = int(opts.in2_size)
in3_size = int(opts.in3_size)
out_size = int(opts.out_size)
trace_size = int(opts.trace_size)

module = my_scale_shift(dev, in1_size, in2_size, in3_size, out_size, trace_size)
# Print the MLIR module to stdout
print(module)
