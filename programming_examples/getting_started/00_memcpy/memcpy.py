# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Multi-column memcpy microbenchmark.

Uses every shim DMA in-out pair on the device to saturate DDR bandwidth.
"""

import sys

import numpy as np

import aie.iron as iron
from aie.iron import Compile, In, Out, kernels, ObjectFifo, Program, Runtime, Worker
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.utils.benchmark import run_iters


@iron.jit
def my_memcpy(
    input0: In,
    output: Out,
    *,
    size: Compile[int],
    xfr_dtype: Compile[type] = np.int32,
):
    num_channels = 2
    device = iron.get_current_device()
    num_columns = device.cols

    if num_channels < 1 or num_channels > 2:
        raise ValueError("Number of channels must be 1 or 2")
    if ((size % 1024) % num_columns % num_channels) != 0:
        raise ValueError(
            f"transfer size ({size}) must be a multiple of 1024 and divisible "
            f"by {num_columns} columns and {num_channels} channels per column"
        )

    line_size = 1024
    line_type = np.ndarray[(line_size,), np.dtype[xfr_dtype]]
    transfer_type = np.ndarray[(size,), np.dtype[xfr_dtype]]
    chunk = size // num_columns // num_channels

    of_ins = [
        ObjectFifo(line_type, name=f"in{i}_{j}")
        for i in range(num_columns)
        for j in range(num_channels)
    ]
    of_outs = [
        ObjectFifo(line_type, name=f"out{i}_{j}")
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    passthrough_fn = kernels.passthrough(tile_size=line_size, dtype=xfr_dtype)

    def core_fn(of_in, of_out, passThroughLine):
        elemOut = of_out.acquire(1)
        elemIn = of_in.acquire(1)
        passThroughLine(elemIn, elemOut, line_size)
        of_in.release(1)
        of_out.release(1)

    my_workers = [
        Worker(
            core_fn,
            [
                of_ins[i * num_channels + j].cons(),
                of_outs[i * num_channels + j].prod(),
                passthrough_fn,
            ],
        )
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    # One TensorAccessPattern per channel. See programming_guide/section-2/section-2c/
    # for a full explanation of n-dimensional data layout transformations. Here:
    #   tensor_dims (1, size)  : logical shape of the full transfer buffer
    #   offset                 : start element index of this channel's chunk
    #   sizes  [1, 1, 1, chunk]: single 1-D transfer of `chunk` elements
    #   strides [0, 0, 0, 1]   : contiguous (stride-1)
    taps = [
        TensorAccessPattern(
            (1, size),
            chunk * i * num_channels + chunk * j,
            [1, 1, 1, chunk],
            [0, 0, 0, 1],
        )
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    rt = Runtime()
    with rt.sequence(transfer_type, transfer_type) as (a_in, b_out):
        rt.start(*my_workers)
        tg = rt.task_group()
        for i in range(num_columns):
            for j in range(num_channels):
                rt.fill(
                    of_ins[i * num_channels + j].prod(),
                    a_in,
                    taps[i * num_channels + j],
                    task_group=tg,
                )
        for i in range(num_columns):
            for j in range(num_channels):
                rt.drain(
                    of_outs[i * num_channels + j].cons(),
                    b_out,
                    taps[i * num_channels + j],
                    wait=True,
                    task_group=tg,
                )
        rt.finish_task_group(tg)

    return Program(device, rt).resolve_program()


def main():
    # Transfer size must be a multiple of 1024 and divisible by columns*channels.
    length = 16777216
    element_type = np.int32

    input0 = iron.arange(length, dtype=element_type, device="npu")
    output = iron.zeros_like(input0)

    # Warm up once so the JIT cache is hot, then time `iters` invocations.
    bench = run_iters(
        my_memcpy,
        input0,
        output,
        size=length,
        xfr_dtype=element_type,
        warmup=1,
        iters=5,
    )

    total_bytes = 2.0 * length * np.dtype(element_type).itemsize  # input + output
    bandwidth_GBps = total_bytes / bench.npu.avg_us / 1e3  # (bytes / µs) → GB/s

    print(
        f"NPU time     (avg/min/max us): "
        f"{bench.npu.avg_us:.1f} / {bench.npu.min_us:.1f} / {bench.npu.max_us:.1f}"
    )
    print(
        f"End-to-end   (avg/min/max us): "
        f"{bench.e2e.avg_us:.1f} / {bench.e2e.min_us:.1f} / {bench.e2e.max_us:.1f}"
    )
    print(f"Effective bandwidth (NPU avg): {bandwidth_GBps:.2f} GB/s")

    e = np.equal(input0.numpy(), output.numpy())
    errors = np.size(e) - np.count_nonzero(e)
    if errors:
        print(f"\nFAIL: {errors} mismatches")
        sys.exit(1)
    print("\nPASS!")


if __name__ == "__main__":
    main()
