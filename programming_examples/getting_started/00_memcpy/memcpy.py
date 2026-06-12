# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Multi-column memcpy microbenchmark.

Uses every shim DMA in-out pair on the device to saturate DDR bandwidth.
The design body is a one-liner delegating to
``iron.algorithms.transform_parallel_typed`` with ``num_channels=2``; the
per-tile kernel is the library passthrough.
"""

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, Out, kernels
from aie.iron.algorithms import transform_parallel_typed
from aie.utils.benchmark import run_iters
from aie.utils.verify import assert_pass


@iron.jit
def my_memcpy(
    input0: In,
    output: Out,
    *,
    size: CompileTime[int],
    xfr_dtype: CompileTime[type] = np.int32,
):
    return transform_parallel_typed(
        kernels.passthrough(tile_size=1024, dtype=xfr_dtype),
        np.ndarray[(size,), np.dtype[xfr_dtype]],
        tile_size=1024,
        num_channels=2,
        pass_size_to_kernel=True,
    )


def main():
    # Transfer size must be a multiple of 1024 × num_columns × num_channels(=2).
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

    assert_pass(output.numpy(), input0.numpy(), fail_msg="memcpy output mismatch")


if __name__ == "__main__":
    main()
