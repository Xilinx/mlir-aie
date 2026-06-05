# multi_column_designs/row_wise_vector_reduce_max.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Multi-column vector reduce-max (row-wise) — IRON + ``@iron.jit``.

8 cores spread across columns each compute a partial max from their
input chunk; a designated reducer core (per-column-group) collects
neighbor partials via ObjectFifos and emits the final reduction.

Library kernels: ``reduce_max(vectorized=True)`` + ``compute_max``,
sharing one ``reduce_max.cc.o`` from the iron kernel library.

Two invocation modes:

  * standalone:   ``python3 row_wise_vector_reduce_max.py``
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
from aie.helpers.taplib.tensortiler2d import TensorTiler2D
from aie.utils.hostruntime.argparse import add_compile_args, add_trace_arg
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass


@iron.jit
def vector_reduce_max(
    a_in: In,
    c_out: Out,
    *,
    in1_size: Compile[int] = 524288,
    out_size: Compile[int] = 4,
    dtype_str: Compile[str] = "i32",
    trace_size: Compile[int] = 0,
):
    if out_size != 4:
        raise ValueError("Output buffer must be size 4 (4 bytes = 1 integer).")

    n_cores = 8
    elems_per_core = 256
    n_channels = n_cores
    if n_cores > 8:
        raise ValueError("This design does not support more than 8 cores.")

    dtype = str_to_dtype(dtype_str)
    in_tensor_size = in1_size // dtype(0).nbytes
    out_tensor_size = out_size // dtype(0).nbytes
    N_per_channel = in_tensor_size // n_channels
    num_iter = in_tensor_size // (elems_per_core * n_channels)

    enable_trace = 1 if trace_size > 0 else 0

    in_ty = np.ndarray[(in_tensor_size,), np.dtype[dtype]]
    op_ty = np.ndarray[(elems_per_core,), np.dtype[dtype]]
    out_ty = np.ndarray[(out_tensor_size,), np.dtype[dtype]]

    in_fifos = []
    out_fifos = []
    for i in range(n_cores):
        in_fifos.append(ObjectFifo(op_ty, name=f"memA{i}"))
        out_fifos.append(ObjectFifo(out_ty, name=f"memC{i}"))

    reduce_max_vector = kernels.reduce_max(tile_size=elems_per_core, dtype=dtype)
    compute_max = kernels.compute_max(dtype=dtype)

    min_val = (
        np.array([bfloat16(float("-inf"))], dtype=dtype)
        if dtype_str == "bf16"
        else np.array([np.iinfo(dtype).min], dtype=dtype)
    )
    nextC_buffers = []
    tmp_buffers = []
    for i in range(n_cores):
        nextC_buffers.append(
            Buffer(
                type=np.ndarray[(out_tensor_size,), np.dtype[dtype]],
                initial_value=min_val,
            )
        )
        tmp_buffers.append(
            Buffer(
                type=np.ndarray[(out_tensor_size,), np.dtype[dtype]],
                initial_value=min_val,
            )
        )

    # One TAP per channel — each reads a contiguous ``N_per_channel``
    # slice of the input tensor.
    taps = TensorTiler2D.simple_tiler((1, in_tensor_size), (1, N_per_channel))

    def core_body(*args):
        compute_max = args[-1]
        reduce_max_vector = args[-2]
        nextC_buffer = args[-3]
        tmp_buffer = args[-4]

        of_in = args[0]
        of_out = args[1]
        in_fifos = args[2:-4]

        for _ in range_(num_iter):
            elem_in = of_in.acquire(1)
            reduce_max_vector(elem_in, tmp_buffer, elems_per_core)
            compute_max(nextC_buffer, tmp_buffer, nextC_buffer)
            of_in.release(1)
        elem_out = of_out.acquire(1)

        if in_fifos:
            inputs = []
            for fifo in in_fifos:
                inputs.append(fifo.acquire(1))

            for elem in inputs[:-1]:
                compute_max(elem, nextC_buffer, nextC_buffer)
            compute_max(inputs[-1], nextC_buffer, elem_out)

            for fifo in in_fifos:
                fifo.release(1)
        else:
            elem_out[0] = nextC_buffer[0]

        of_out.release(1)

    workers = []
    for i in range(n_cores):
        fifo_args = [in_fifos[i].cons(), out_fifos[i].prod()]
        if (
            (i == 1 and n_cores >= 2)
            or (i == 4 and n_cores == 5)
            or (i == 5 and n_cores > 5)
        ):
            if i == 1:
                cores_per_col = min(4, n_cores)
                fifo_args.append(out_fifos[0].cons())
                for j in range(2, cores_per_col):
                    fifo_args.append(out_fifos[j].cons())
            else:
                fifo_args.append(out_fifos[1].cons())
                if i == 5:
                    fifo_args.append(out_fifos[4].cons())
                    fifo_args.extend(out_fifos[j].cons() for j in range(6, n_cores))

        fifo_args.extend(
            [tmp_buffers[i], nextC_buffers[i], reduce_max_vector, compute_max]
        )

        workers.append(Worker(core_body, fn_args=fifo_args, trace=enable_trace))

    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (a, c):
        if trace_size > 0:
            rt.enable_trace(trace_size)
        rt.start(*workers)
        for i in range(n_channels):
            rt.fill(in_fifos[i].prod(), a, taps[i])
        rt.drain(
            out_fifos[
                0 if n_cores == 1 else 1 if n_cores < 5 else 4 if n_cores == 5 else 5
            ].cons(),
            c,
            wait=True,
        )

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Multi-Column Vector Reduce Max (row-wise)")
    add_compile_args(p)
    p.add_argument("-i1s", "--in1_size", type=int, default=524288, help="bytes")
    p.add_argument("-os", "--out_size", type=int, default=4, help="bytes (always 4)")
    p.add_argument("-dt", "--dtype", type=str, default="i32", choices=["i32", "bf16"])
    add_trace_arg(p)
    return p


def _compile_kwargs(opts):
    return dict(
        in1_size=opts.in1_size,
        out_size=opts.out_size,
        dtype_str=opts.dtype,
        trace_size=opts.trace_size,
    )


def _run_and_verify(opts):
    dtype = str_to_dtype(opts.dtype)
    num_elements = opts.in1_size // dtype(0).nbytes
    out_num_elements = opts.out_size // dtype(0).nbytes

    rng = np.random.default_rng(0)
    if opts.dtype == "i32":
        in_np = rng.integers(-1000, 1000, size=(num_elements,), dtype=np.int32)
    else:
        in_np = rng.uniform(-1000.0, 1000.0, size=(num_elements,)).astype(dtype)
    in_t = iron.tensor(in_np, dtype=dtype, device="npu")
    out_t = iron.zeros(out_num_elements, dtype=dtype, device="npu")

    vector_reduce_max(in_t, out_t, **_compile_kwargs(opts))

    expected_max = in_np.max()
    actual_max = out_t.numpy()[0]
    assert_pass(
        actual_max, expected_max, fail_msg=f"expected {expected_max}, got {actual_max}"
    )


def _validate(opts):
    if opts.in1_size % 64 != 0 or opts.in1_size < 512:
        sys.exit(f"in1_size ({opts.in1_size}) must be a multiple of 64 and >= 512")


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        vector_reduce_max,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        validate=_validate,
    )


if __name__ == "__main__":
    main()
