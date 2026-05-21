# multi_column_designs/col_wise_vector_reduce_max.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Multi-column vector reduce-max (col-wise) — Iron + ``@iron.jit``.

Spread the reduction across multiple columns (default 8 on NPU1, up to
16 on NPU2): each core computes a partial max over its chunk, then the
last core in each column collects partials from neighboring cores via
ObjectFifos and emits the final per-column reduction; the last core
overall produces the final output.

Library kernels: ``reduce_max(vectorized=True)`` + ``compute_max``; both
share one ``reduce_max.cc.o`` via the iron kernel library.

Two invocation modes:

  * standalone:   ``python3 col_wise_vector_reduce_max.py``
  * compile-only: ``... --xclbin-path=PATH --insts-path=PATH``  (Makefile)
"""

import argparse
import sys

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (
    Buffer,
    Compile,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    kernels,
    str_to_dtype,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.utils.hostruntime import set_current_device


def _device_for(dev_str):
    return NPU1() if dev_str == "npu" else NPU2()


# cores_per_col=2 is baked into the neighbor-FIFO wiring below; pass the
# matching aiecc flag so the placer respects the 2-cores-per-column layout
# on any column count (NPU1: 4 cols × 2 = 8, NPU2: 8 cols × 2 = 16).
@iron.jit(aiecc_flags=["--cores-per-col", "2"])
def vector_reduce_max(
    a_in: In,
    c_out: Out,
    *,
    in1_size: Compile[int] = 524288,
    out_size: Compile[int] = 4,
    num_cores: Compile[int] = 8,
    dtype_str: Compile[str] = "i32",
    trace_size: Compile[int] = 0,
):
    if out_size != 4:
        raise ValueError("Output buffer must be size 4 (4 bytes = 1 integer).")

    enable_trace = 1 if trace_size > 0 else None
    cores_per_col = 2

    dtype = str_to_dtype(dtype_str)
    in_num_elements = in1_size // dtype(0).nbytes
    out_num_elements = out_size // dtype(0).nbytes

    chunk = in_num_elements // num_cores
    tile_size = chunk if chunk < 4096 else 4096
    N_div_n = in_num_elements // (tile_size * num_cores)

    in_tensor_ty = np.ndarray[(in_num_elements,), np.dtype[dtype]]
    out_tensor_ty = np.ndarray[(out_num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(tile_size,), np.dtype[dtype]]
    fifodepth = 2

    of_in1s = [
        ObjectFifo(tile_ty, name=f"in1_{i}", depth=fifodepth) for i in range(num_cores)
    ]
    of_outs = [
        ObjectFifo(out_tensor_ty, name=f"out_{i}", depth=fifodepth)
        for i in range(num_cores)
    ]

    reduce_max_vector = kernels.reduce_max(tile_size=tile_size, dtype=dtype)
    compute_max = kernels.compute_max(dtype=dtype)

    min_val = (
        np.array([bfloat16(float("-inf"))], dtype=dtype)
        if dtype_str == "bf16"
        else np.array([np.iinfo(dtype).min], dtype=dtype)
    )
    nextC_buffers = []
    tmp_buffers = []
    for i in range(num_cores):
        nextC_buffers.append(
            Buffer(
                type=np.ndarray[(out_num_elements,), np.dtype[dtype]],
                initial_value=min_val,
            )
        )
        tmp_buffers.append(
            Buffer(
                type=np.ndarray[(out_num_elements,), np.dtype[dtype]],
                initial_value=min_val,
            )
        )

    def core_body(*args):
        compute_max = args[-1]
        reduce_max_vector = args[-2]
        tmp_buffer = args[-3]
        c_buffer = args[-4]

        of_in1 = args[0]
        of_out = args[1]
        neighbor_of_in1s = args[2:-4]

        for _ in range_(N_div_n):
            elem_in1 = of_in1.acquire(1)
            reduce_max_vector(elem_in1, tmp_buffer, tile_size)
            compute_max(c_buffer, tmp_buffer, c_buffer)
            of_in1.release(1)

        elem_out = of_out.acquire(1)
        if neighbor_of_in1s:
            elem_in1s = []
            for neighbor_of in neighbor_of_in1s:
                elem_in1s.append(neighbor_of.acquire(1))

            for elem in elem_in1s[:-1]:
                compute_max(elem, c_buffer, c_buffer)
            compute_max(elem_in1s[-1], c_buffer, elem_out)

            for neighbor_of in neighbor_of_in1s:
                neighbor_of.release(1)
        else:
            elem_out[0] = c_buffer[0]
        of_out.release(1)

    my_workers = []
    for i in range(num_cores):
        fifo_args = [of_in1s[i].cons(), of_outs[i].prod()]
        if cores_per_col - 1 < i:
            fifo_args.append(of_outs[i - cores_per_col].cons())
            if num_cores - cores_per_col < i:
                fifo_args.append(of_outs[i - 1].cons())

        fifo_args.extend(
            [nextC_buffers[i], tmp_buffers[i], reduce_max_vector, compute_max]
        )
        my_workers.append(
            Worker(core_body, fn_args=fifo_args, trace=enable_trace)
        )

    taps = [
        TensorAccessPattern(
            (1, in_num_elements),
            chunk * i,
            [1, 1, 1, chunk],
            [0, 0, 0, 1],
        )
        for i in range(num_cores)
    ]

    rt = Runtime()
    with rt.sequence(in_tensor_ty, out_tensor_ty) as (a, c):
        if enable_trace:
            rt.enable_trace(trace_size)
        rt.start(*my_workers)
        for i in range(num_cores):
            rt.fill(of_in1s[i].prod(), a, taps[i])
        rt.drain(of_outs[num_cores - 1].cons(), c, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Multi-Column Vector Reduce Max (col-wise)")
    p.add_argument("-d", "--dev", type=str, choices=["npu", "npu2"], default="npu")
    p.add_argument("-i1s", "--in1_size", type=int, default=524288, help="bytes")
    p.add_argument("-os", "--out_size", type=int, default=4, help="bytes (always 4)")
    p.add_argument("-nc", "--num_cores", type=int, default=8)
    p.add_argument("-dt", "--dtype", type=str, default="i32", choices=["i32", "bf16"])
    p.add_argument("-t", "--trace_size", type=int, default=0)
    p.add_argument("--xclbin-path", type=str, default=None)
    p.add_argument("--insts-path", type=str, default=None)
    return p


def _validate(opts):
    max_cores = 8 if opts.dev == "npu" else 16
    if opts.num_cores > max_cores:
        sys.exit(f"--num_cores ({opts.num_cores}) exceeds {opts.dev} max ({max_cores})")
    if (
        opts.in1_size % 64 != 0
        or opts.in1_size < 64 * opts.num_cores
        or opts.in1_size % opts.num_cores != 0
    ):
        sys.exit(
            f"in1_size ({opts.in1_size}) must be a multiple of 64 and {opts.num_cores}, "
            f"and >= {64 * opts.num_cores}"
        )


def _compile_kwargs(opts):
    return dict(
        in1_size=opts.in1_size,
        out_size=opts.out_size,
        num_cores=opts.num_cores,
        dtype_str=opts.dtype,
        trace_size=opts.trace_size,
    )


def _compile_only(opts):
    if not opts.insts_path:
        sys.exit("--xclbin-path requires --insts-path (must be set together)")
    set_current_device(_device_for(opts.dev))
    spec = vector_reduce_max.specialize(**_compile_kwargs(opts))
    spec.compile(xclbin_path=opts.xclbin_path, inst_path=opts.insts_path)


def _run_and_verify(opts):
    dtype = str_to_dtype(opts.dtype)
    num_elements = opts.in1_size // dtype(0).nbytes
    out_num_elements = opts.out_size // dtype(0).nbytes

    rng = np.random.default_rng(0)
    if opts.dtype == "i32":
        in_np = rng.integers(-1000, 1000, size=(num_elements,), dtype=np.int32)
    else:
        in_np = rng.uniform(-1000.0, 1000.0, size=(num_elements,)).astype(dtype)
    out_np = np.zeros((out_num_elements,), dtype=dtype)

    in_t = iron.tensor(in_np, dtype=dtype, device="npu")
    out_t = iron.tensor(out_np, dtype=dtype, device="npu")

    vector_reduce_max(in_t, out_t, **_compile_kwargs(opts))

    expected_max = in_np.max()
    actual_max = out_t.numpy()[0]
    if actual_max != expected_max:
        sys.exit(f"FAIL! expected {expected_max}, got {actual_max}")
    print("PASS!")


def main():
    opts = _make_argparser().parse_args()
    _validate(opts)
    if opts.xclbin_path:
        _compile_only(opts)
        return
    _run_and_verify(opts)


if __name__ == "__main__":
    main()
