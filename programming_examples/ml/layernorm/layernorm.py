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
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib.tap import TensorAccessPattern

# from ml_dtypes import float


def layernorm(dev, rows, cols, trace_size):
    # enable_trace = 1 if trace_size > 0 else None

    n_cores = 1

    total_volume = rows * cols

    dtype = np.ndarray[(total_volume,), np.dtype[np.float32]]

    cols_per_core = cols // n_cores  # Keep it 16
    chunk_volume = rows * cols_per_core
    chunk_type = np.ndarray[(chunk_volume,), np.dtype[np.float32]]

    of_in = [ObjectFifo(chunk_type, name=f"in_{i}") for i in range(n_cores)]
    of_out = [ObjectFifo(chunk_type, name=f"out_{i}") for i in range(n_cores)]

    layer_norm_kernel = Kernel(
        "layer_norm", "layer_norm.o", [chunk_type, chunk_type, np.int32, np.int32]
    )
    taps_in = []
    taps_out = []
    for i in range(n_cores):
        taps = TensorAccessPattern(
            (rows, cols),
            offset=rows * cols_per_core * i,
            sizes=[1, 1, rows, cols_per_core],
            strides=[0, 0, 1, rows],
        )
        # if i == 0:
        #     taps.visualize(
        #         title=f"Core {i} input tap",
        #         show_arrows=True,
        #         plot_access_count=True,
        #         file_path=f"core_{i}_input_tap.png",
        #     )
        taps_in.append(taps)

    for i in range(n_cores):
        taps = TensorAccessPattern(
            (rows, cols),
            offset=rows * cols_per_core * i,
            sizes=[1, 1, rows, cols_per_core],
            strides=[0, 0, 1, rows],
        )
        # taps.visualize(
        #     title=f"Core {i} output tap",
        #     show_arrows=True,
        #     plot_access_count=True,
        #     file_path=f"core_{i}_output_tap.png",
        # )
        taps_out.append(taps)

    def core_body(of_in, of_out, layer_norm_kernel):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        layer_norm_kernel(elem_in, elem_out, rows, cols_per_core)
        of_in.release(1)
        of_out.release(1)

    workers = [
        Worker(
            core_body,
            fn_args=[of_in[i].cons(), of_out[i].prod(), layer_norm_kernel],
            trace=None,
            # trace=enable_trace,
        )
        for i in range(n_cores)
    ]

    rt = Runtime()
    with rt.sequence(dtype, dtype) as (a_in, c_out):
        rt.enable_trace(trace_size)
        rt.start(*workers)
        for i in range(n_cores):
            rt.fill(of_in[i].prod(), a_in, taps_in[i])
        for i in range(n_cores):
            rt.drain(of_out[i].cons(), c_out, taps_out[i], wait=True)
    return Program(dev, rt).resolve_program(SequentialPlacer())


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
p.add_argument("-r", "--rows", required=True, dest="rows", help="Row size")
p.add_argument("-c", "--cols", required=True, dest="cols", help="Col size")
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
    dev = NPU1()
elif opts.device == "npu2":
    dev = NPU2()
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(opts.device))

rows = int(opts.rows)
cols = int(opts.cols)
trace_size = int(opts.trace_size)

print(layernorm(dev, rows, cols, trace_size))
