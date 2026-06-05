# single_column_designs/vector_reduce_max_shared.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Single-column vector reduce-max (shared variant) — IRON API + ``@iron.jit``.

4 cores compute partial max in parallel from a shared input split across
them; core 1 collects the other 3 cores' partials via shared ObjectFifos
and produces the final reduction.

Kernels come from the iron kernel library (``iron.kernels.reduce_max``,
``iron.kernels.compute_max``); both factories point at the same
``reduce_max.cc`` source and share one compiled ``.o``.

Two invocation modes:

  * standalone:   ``python3 vector_reduce_max_shared.py``
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
from aie.helpers.util import np_ndarray_type_get_shape
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

    n_cores = 4
    n_mem_elems = 2048
    elems_per_core = n_mem_elems // n_cores

    dtype = str_to_dtype(dtype_str)
    in_tensor_size = in1_size // dtype(0).nbytes
    out_tensor_size = out_size // dtype(0).nbytes
    num_iter = in_tensor_size // n_mem_elems

    enable_trace = 1 if trace_size > 0 else 0

    in_ty = np.ndarray[(in_tensor_size,), np.dtype[dtype]]
    mem_ty = np.ndarray[(n_mem_elems,), np.dtype[dtype]]
    op_ty = np.ndarray[(elems_per_core,), np.dtype[dtype]]
    out_ty = np.ndarray[(out_tensor_size,), np.dtype[dtype]]

    of_in = ObjectFifo(mem_ty, name="of_in")
    of_a_offsets = [
        (np.prod(np_ndarray_type_get_shape(mem_ty)) // n_cores) * i
        for i in range(n_cores)
    ]
    in_fifos = of_in.cons().split(
        of_a_offsets,
        obj_types=[op_ty] * n_cores,
        names=[f"memA{i}" for i in range(n_cores)],
    )

    min_val = (
        np.array([bfloat16(float("-inf"))], dtype=dtype)
        if dtype_str == "bf16"
        else np.array([np.iinfo(dtype).min], dtype=dtype)
    )
    out_fifos = []
    nextC_buffers = []
    tmp_buffers = []
    for i in range(n_cores):
        out_fifos.append(ObjectFifo(out_ty, name=f"memC{i}"))
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

    # Library kernels: both come from reduce_max.cc, sharing one .o.
    reduce_max_vector = kernels.reduce_max(tile_size=elems_per_core, dtype=dtype)
    compute_max = kernels.compute_max(dtype=dtype)

    def start_core_body(
        of_in, of_out, nextC_buffer, tmp_buffer, reduce_max_vector, compute_max
    ):
        elem_out = of_out.acquire(1)
        for _ in range_(num_iter):
            elem_in = of_in.acquire(1)
            reduce_max_vector(elem_in, tmp_buffer, elems_per_core)
            compute_max(nextC_buffer, tmp_buffer, nextC_buffer)
            of_in.release(1)
        elem_out[0] = nextC_buffer[0]
        of_out.release(1)

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

        inputs = []
        for fifo in in_fifos:
            inputs.append(fifo.acquire(1))

        for elem in inputs[:-1]:
            compute_max(elem, nextC_buffer, nextC_buffer)
        compute_max(inputs[-1], nextC_buffer, elem_out)

        for fifo in in_fifos:
            fifo.release(1)
        of_out.release(1)

    workers = []
    for i in range(n_cores):
        if i == 1:
            fifo_args = [in_fifos[i].cons(), out_fifos[i].prod()]
            for j in range(n_cores - 1):
                if j < i:
                    fifo_args.append(out_fifos[j].cons())
                else:
                    fifo_args.append(out_fifos[j + 1].cons())
            fifo_args.extend(
                [nextC_buffers[i], tmp_buffers[i], reduce_max_vector, compute_max]
            )
            workers.append(Worker(core_body, fn_args=fifo_args, trace=enable_trace))
        else:
            workers.append(
                Worker(
                    start_core_body,
                    fn_args=[
                        in_fifos[i].cons(),
                        out_fifos[i].prod(),
                        nextC_buffers[i],
                        tmp_buffers[i],
                        reduce_max_vector,
                        compute_max,
                    ],
                    trace=enable_trace,
                )
            )

    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (a, c):
        if trace_size > 0:
            rt.enable_trace(trace_size)
        rt.start(*workers)
        rt.fill(of_in.prod(), a)
        rt.drain(out_fifos[1].cons(), c, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Single-Column Vector Reduce Max (shared)")
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
