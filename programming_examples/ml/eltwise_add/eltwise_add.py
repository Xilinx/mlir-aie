#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.
from ml_dtypes import bfloat16
import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2
from aie.iron.controlflow import range_
from aie.helpers.util import np_ndarray_type_get_shape


def my_eltwise_add(dev, in1_size, in2_size, out_size, trace_size):
    in1_dtype = bfloat16
    in2_dtype = bfloat16
    out_dtype = bfloat16

    tensor_size = in1_size // in1_dtype(0).nbytes

    # Tile sizes
    tile_size = 1024
    tensor_div_tile = tensor_size // tile_size

    n_cores = 2
    tiles = tensor_div_tile // n_cores

    assert in2_size == in1_size, "input2 buffer size must match input1 buffer size."
    assert out_size == in1_size, "Output buffer size must match input1 buffer size."

    enable_trace = 1 if trace_size > 0 else 0

    tensor_ty = np.ndarray[(tensor_size,), np.dtype[out_dtype]]
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
    eltwise_add_bf16_vector = Kernel(
        "eltwise_add_bf16_vector", "add.o", [tile_ty, tile_ty, tile_ty]
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

    # Task for the cores to perform
    def core_fn(of_a, of_b, of_c, eltwise_add):
        for _ in range_(tiles):
            elem_out = of_c.acquire(1)
            elem_in_a = of_a.acquire(1)
            elem_in_b = of_b.acquire(1)
            eltwise_add(elem_in_a, elem_in_b, elem_out)
            of_a.release(1)
            of_b.release(1)
            of_c.release(1)

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
                    eltwise_add_bf16_vector,
                ],
                # trace=enable_trace,
            )
        )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (A, B, C):
        rt.enable_trace(trace_size, workers=[workers[0]])
        rt.start(*workers)
        rt.fill(inA.prod(), A)
        rt.fill(inB.prod(), B)
        rt.drain(outC.cons(), C, wait=True)

    # Place components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())

p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
p.add_argument(
    "-i1s", "--in1_size", required=True, dest="in1_size", help="Input 1 size"
)
p.add_argument(
    "-i2s", "--in2_size", required=True, dest="in2_size", help="Input 2 size"
)
p.add_argument("-os", "--out_size", required=True, dest="out_size", help="Output size")
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
    dev = NPU2()
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(opts.device))

in1_size = int(opts.in1_size)
in2_size = int(opts.in2_size)
out_size = int(opts.out_size)
trace_size = int(opts.trace_size)

module = my_eltwise_add(dev, in1_size, in2_size, out_size, trace_size)
print(module)

