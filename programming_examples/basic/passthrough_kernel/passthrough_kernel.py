# passthrough_kernel/passthrough_kernel.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Passthrough kernel — Iron API + ``@iron.jit`` variant."""

import argparse
import sys
import time

import numpy as np

import aie.iron as iron
from aie.iron import (
    Compile,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    kernels,
)
from aie.iron.controlflow import range_
from aie.utils.trace import TraceConfig
from aie.utils.trace.utils import print_cycles_summary


_TRACE_JSON = "trace_passthrough_kernel.json"


@iron.jit
def my_passthrough_kernel(
    in_tensor: In,
    out_tensor: Out,
    *,
    n: Compile[int],
    trace_config: Compile[TraceConfig | None] = None,
):
    in1_dtype = np.uint8
    line_size = n // 4
    line_type = np.ndarray[(line_size,), np.dtype[in1_dtype]]
    vector_type = np.ndarray[(n,), np.dtype[in1_dtype]]

    of_in = ObjectFifo(line_type, name="in")
    of_out = ObjectFifo(line_type, name="out")

    pass_through_line = kernels.passthrough(tile_size=line_size, dtype=in1_dtype)

    def core_fn(of_in, of_out, pass_through_line):
        for _ in range_(sys.maxsize):
            elem_out = of_out.acquire(1)
            elem_in = of_in.acquire(1)
            pass_through_line(elem_in, elem_out, line_size)
            of_in.release(1)
            of_out.release(1)

    worker = Worker(
        core_fn,
        [of_in.cons(), of_out.prod(), pass_through_line],
        trace=1 if trace_config else 0,
    )

    rt = Runtime()
    with rt.sequence(vector_type, vector_type, vector_type) as (a_in, b_out, _):
        if trace_config:
            rt.enable_trace(trace_config.trace_size, workers=[worker])
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), b_out, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dev", default="npu", help="AIE device (npu|npu2)")
    p.add_argument(
        "-i1s",
        "--in1_size",
        type=int,
        default=4096,
        help="Input buffer size in bytes (uint8 elements)",
    )
    p.add_argument(
        "-os",
        "--out_size",
        type=int,
        default=4096,
        help="Output buffer size in bytes (must equal --in1_size)",
    )
    p.add_argument(
        "-t",
        "--trace_size",
        type=int,
        default=0,
        help="Trace buffer size in bytes (0 disables tracing)",
    )
    p.add_argument("-w", "--warmup", type=int, default=10)
    p.add_argument("-n", "--iters", type=int, default=20)
    opts = p.parse_args()

    in1_size = opts.in1_size
    out_size = opts.out_size
    trace_size = opts.trace_size

    if in1_size % 64 != 0 or in1_size < 512:
        sys.exit(f"in1_size ({in1_size}) must be a multiple of 64 and >= 512")
    assert out_size == in1_size, "out_size must equal in1_size"

    in1_dtype = np.uint8
    n_elems = in1_size // np.dtype(in1_dtype).itemsize

    in_tensor = iron.tensor(
        np.arange(0, n_elems, dtype=in1_dtype), dtype=in1_dtype, device=opts.dev
    )
    out_tensor = iron.zeros([n_elems], dtype=in1_dtype, device=opts.dev)

    trace_config = TraceConfig(trace_size=trace_size) if trace_size > 0 else None

    if trace_config is not None:
        # trace.txt is overwritten each call, so only one iteration is meaningful
        warmup, iters = 0, 1
    else:
        warmup, iters = opts.warmup, opts.iters

    e2e_total = npu_total = 0.0
    e2e_max = npu_max = 0.0
    e2e_min = npu_min = float("inf")
    for i in range(warmup + iters):
        start = time.perf_counter()
        _, result = my_passthrough_kernel(
            in_tensor, out_tensor, n=n_elems, trace_config=trace_config
        )
        e2e_us = (time.perf_counter() - start) * 1_000_000
        npu_us = result.npu_time / 1_000.0
        if i >= warmup:
            e2e_total += e2e_us
            e2e_min = min(e2e_min, e2e_us)
            e2e_max = max(e2e_max, e2e_us)
            npu_total += npu_us
            npu_min = min(npu_min, npu_us)
            npu_max = max(npu_max, npu_us)

    expected = in_tensor.numpy()
    computed = out_tensor.numpy()
    if not np.array_equal(expected, computed):
        mismatches = int(np.sum(expected != computed))
        print(f"FAIL! {mismatches} mismatches out of {expected.size}")
        sys.exit(1)

    if iters > 0:
        print(
            f"\nNPU time     (avg/min/max us): "
            f"{npu_total / iters:.1f} / {npu_min:.1f} / {npu_max:.1f}"
        )
        print(
            f"End-to-end   (avg/min/max us): "
            f"{e2e_total / iters:.1f} / {e2e_min:.1f} / {e2e_max:.1f}"
        )

    if trace_config is not None:
        if trace_config.physical_mlir_path is None:
            sys.exit(
                "trace requested but physical_mlir_path was not set by the JIT "
                "runtime — cannot parse trace events."
            )
        trace_config.trace_to_json(
            trace_config.physical_mlir_path, output_name=_TRACE_JSON
        )
        print_cycles_summary(_TRACE_JSON)

    print("PASS!")


if __name__ == "__main__":
    main()
