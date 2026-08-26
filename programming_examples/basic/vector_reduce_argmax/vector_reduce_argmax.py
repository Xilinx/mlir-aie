# vector_reduce_argmax/vector_reduce_argmax.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Vector reduce-argmax, chained across one column — IRON API + ``@iron.jit``.

The argmax counterpart of ``../vector_reduce_max/single_column_designs``:
the input streams through a memtile buffer that is split across 4 cores,
each core folds its share into a running (value, index) record, and the
records fold pairwise down the column so core 0 emits the single winner.
The host reads 8 bytes whatever the input length.

Every record leaves a core carrying a GLOBAL index -- the kernel adds the
``index_offset`` argument, which is the slice's start -- so the fold is
order-independent and needs no per-core fixup on the host.

Kernels come from the iron kernel library (``iron.kernels.argmax``,
``iron.kernels.argmax_combine``); both factories share one ``.o``.

Two invocation modes:

  * standalone:   ``python3 vector_reduce_argmax.py``
  * compile-only: ``... --xclbin-path=PATH --insts-path=PATH``
"""

import argparse
import sys

import aie.iron as iron
import numpy as np
from aie.iron import (
    Buffer,
    CompileTime,
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
from aie.utils.hostruntime.argparse import add_compile_args, add_trace_arg
from aie.utils.hostruntime.cli import run_design_cli

N_CORES = 4
# Elements staged in the memtile per iteration, split N_CORES ways.
N_MEM_ELEMS = 2048
# The record the kernel writes: [value bits, index], both int32.
RECORD_ELEMS = 2
RECORD_BYTES = RECORD_ELEMS * 4


@iron.jit
def vector_reduce_argmax(
    a_in: In,
    c_out: Out,
    *,
    in1_size: CompileTime[int] = 65536,
    out_size: CompileTime[int] = RECORD_BYTES,
    dtype_str: CompileTime[str] = "bf16",
    trace_size: CompileTime[int] = 0,
):
    if out_size != RECORD_BYTES:
        raise ValueError(f"Output buffer must be size {RECORD_BYTES} (value + index).")

    dtype = str_to_dtype(dtype_str)
    in_tensor_size = in1_size // dtype(0).nbytes
    elems_per_core = N_MEM_ELEMS // N_CORES
    num_iter = in_tensor_size // N_MEM_ELEMS

    enable_trace = 1 if trace_size > 0 else 0

    in_ty = np.ndarray[(in_tensor_size,), np.dtype[dtype]]
    mem_ty = np.ndarray[(N_MEM_ELEMS,), np.dtype[dtype]]
    op_ty = np.ndarray[(elems_per_core,), np.dtype[dtype]]
    record_ty = np.ndarray[(RECORD_ELEMS,), np.dtype[np.int32]]

    of_in = ObjectFifo(mem_ty, name="of_in")
    in_fifos = of_in.cons().split(
        [elems_per_core * i for i in range(N_CORES)],
        obj_types=[op_ty] * N_CORES,
        names=[f"memA{i}" for i in range(N_CORES)],
    )
    out_fifos = [ObjectFifo(record_ty, name=f"memC{i}") for i in range(N_CORES)]

    argmax = kernels.argmax(tile_size=elems_per_core, dtype=dtype)
    argmax_combine = kernels.argmax_combine(dtype=dtype)

    zero_record = np.zeros(RECORD_ELEMS, dtype=np.int32)
    partials = [
        Buffer(type=record_ty, initial_value=zero_record) for _ in range(N_CORES)
    ]
    tmps = [Buffer(type=record_ty, initial_value=zero_record) for _ in range(N_CORES)]

    def reduce_slice(of_in, argmax, argmax_combine, partial, tmp, first_index):
        # The first iteration writes `partial` outright rather than folding into
        # it: a core-resident buffer keeps its value from the previous run of the
        # same design, so seeding one with an identity record would make the
        # result depend on run order.
        elem_in = of_in.acquire(1)
        argmax(elem_in, partial, elems_per_core, first_index)
        of_in.release(1)
        next_index = first_index + N_MEM_ELEMS
        for i in range_(num_iter - 1):
            elem_in = of_in.acquire(1)
            argmax(elem_in, tmp, elems_per_core, next_index + i * N_MEM_ELEMS)
            argmax_combine(partial, tmp, partial)
            of_in.release(1)

    # End of the chain: nothing to fold in, so this core's own record goes out.
    def tail_core_body(
        of_in, of_out, argmax, argmax_combine, partial, tmp, first_index
    ):
        reduce_slice(of_in, argmax, argmax_combine, partial, tmp, first_index)
        elem_out = of_out.acquire(1)
        elem_out[0] = partial[0]
        elem_out[1] = partial[1]
        of_out.release(1)

    def core_body(
        of_in, of_out, next_in, argmax, argmax_combine, partial, tmp, first_index
    ):
        reduce_slice(of_in, argmax, argmax_combine, partial, tmp, first_index)
        elem_next = next_in.acquire(1)
        elem_out = of_out.acquire(1)
        argmax_combine(partial, elem_next, elem_out)
        next_in.release(1)
        of_out.release(1)

    workers = []
    for i in range(N_CORES):
        common = [argmax, argmax_combine, partials[i], tmps[i], i * elems_per_core]
        if i == N_CORES - 1:
            fn_args = [in_fifos[i].cons(), out_fifos[i].prod()] + common
            body = tail_core_body
        else:
            fn_args = [
                in_fifos[i].cons(),
                out_fifos[i].prod(),
                out_fifos[i + 1].cons(),
            ] + common
            body = core_body
        workers.append(Worker(body, fn_args=fn_args, trace=enable_trace))

    def sequence(a, c, in_h, out_h):
        in_h.fill(a)
        out_h.drain(c, wait=True)

    rt = Runtime(sequence, [in_ty, record_ty, of_in.prod(), out_fifos[0].cons()])
    prog = Program(iron.get_current_device(), rt, workers=workers)
    if trace_size > 0:
        prog.enable_trace(trace_size)

    return prog.resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Vector Reduce Argmax")
    add_compile_args(p)
    p.add_argument("-i1s", "--in1_size", type=int, default=65536, help="bytes")
    p.add_argument(
        "-os", "--out_size", type=int, default=RECORD_BYTES, help="bytes (always 8)"
    )
    p.add_argument("-dt", "--dtype", type=str, default="bf16", choices=["i32", "bf16"])
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

    rng = np.random.default_rng(0)
    if opts.dtype == "i32":
        in_np = rng.integers(-100000, 100000, size=(num_elements,), dtype=np.int32)
    else:
        in_np = rng.uniform(-1000.0, 1000.0, size=(num_elements,)).astype(dtype)
    in_t = iron.tensor(in_np, dtype=dtype, device="npu")
    out_t = iron.zeros(RECORD_ELEMS, dtype=np.int32, device="npu")

    vector_reduce_argmax(in_t, out_t, **_compile_kwargs(opts))

    actual = out_t.numpy()
    expected = kernels.argmax_ref(in_np)
    if not np.array_equal(actual, expected):
        print(f"FAIL: expected record {expected}, got {actual}")
        sys.exit(1)
    print("PASS!")


def _validate(opts):
    dtype = str_to_dtype(opts.dtype)
    elems = opts.in1_size // dtype(0).nbytes
    if elems % N_MEM_ELEMS != 0:
        sys.exit(
            f"in1_size ({opts.in1_size} bytes = {elems} {opts.dtype}) must be a "
            f"whole number of {N_MEM_ELEMS}-element iterations"
        )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        vector_reduce_argmax,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        validate=_validate,
    )


if __name__ == "__main__":
    main()
