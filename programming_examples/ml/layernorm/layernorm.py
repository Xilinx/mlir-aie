# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2Col1
from ml_dtypes import bfloat16


def layernorm(dev, in_size, trace_size):
    enable_trace = 1 if trace_size > 0 else None

    # Define tensor types
    line_size = in_size // bfloat16(0).nbytes

    dtype = np.ndarray[(line_size,), np.dtype[bfloat16]]

    of_in = ObjectFifo(dtype, name="in")
    of_out = ObjectFifo(dtype, name="out")

    layer_norm_kernel = Kernel("layer_norm", "layer_norm.o", [dtype, dtype, np.int32, np.int32])

    rows = 16
    cols = 64

    def core_body(of_in, of_out, layer_norm_kernel):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        layer_norm_kernel(elem_in, elem_out, rows, cols)
        of_in.release(1)
        of_out.release(1)

    worker = Worker(
        core_body,
        fn_args=[of_in.cons(), of_out.prod(), layer_norm_kernel],
        trace=enable_trace,
    )

    rt = Runtime()
    with rt.sequence(dtype, dtype) as (a_in, c_out):
        rt.enable_trace(enable_trace)
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), c_out, wait=True)
    return Program(dev, rt).resolve_program(SequentialPlacer())


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
p.add_argument("-i1s", "--in_size", required=True, dest="in_size", help="Input size")
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

in_size = int(opts.in_size)
if in_size % 64 != 0 or in_size < 512:
    print(
        "In1 buffer size ("
        + str(in_size)
        + ") must be a multiple of 64 and greater than or equal to 512"
    )
    raise ValueError
trace_size = int(opts.trace_size)

print(layernorm(dev, in_size, trace_size))