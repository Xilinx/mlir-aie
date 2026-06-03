# scale_shift/scale_shift.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Time-multiplexed bf16 scale-and-shift — IRON API design with ``@iron.jit``.

Two cores compute ``D = A * B + C`` in two passes over the same workers:

  1. Phase 1: ``rtp = 1`` (multiply) — workers compute ``D = A * B``.
  2. Phase 2: ``rtp = 0`` (add)      — workers compute ``D = D + C``.

A per-worker ``Buffer(..., use_write_rtp=True)`` carries the runtime
parameter; a ``WorkerRuntimeBarrier`` synchronizes the runtime sequence
with the worker so the new ``rtp`` value is visible before the worker
starts the next phase.
"""

import argparse
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (
    Buffer,
    Compile,
    In,
    Out,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    WorkerRuntimeBarrier,
)
from aie.iron.device import device_from_args
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.helpers.util import np_ndarray_type_get_shape
from aie.utils import config
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

_KERNEL_SRC = Path(__file__).resolve().parents[3] / "aie_kernels/aie2/scale_shift.cc"


def _scale_shift_extern(tile_ty):
    return ExternalFunction(
        "eltwise_mul_add_bf16_vector",
        source_file=str(_KERNEL_SRC),
        arg_types=[tile_ty, tile_ty, tile_ty, np.int32],
        include_dirs=[config.cxx_header_path()],
    )


@iron.jit
def scale_shift(
    a_in: In,
    b_in: In,
    c_in: In,
    d_out: Out,
    *,
    size: Compile[int] = 65536,
):
    device = iron.get_current_device()
    n_cores = 2
    tile_size = 1024

    if size % (tile_size * n_cores) != 0:
        raise ValueError(f"size ({size}) must be a multiple of {tile_size * n_cores}")

    tiles_per_core = size // tile_size // n_cores

    tensor_ty = np.ndarray[(size,), np.dtype[bfloat16]]
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    memtile_ty = np.ndarray[(tile_size * n_cores,), np.dtype[bfloat16]]

    # Split each input across n_cores via a memtile FIFO.
    def _split(of, name):
        offsets = [
            (np.prod(np_ndarray_type_get_shape(memtile_ty)) // n_cores) * i
            for i in range(n_cores)
        ]
        return of.cons().split(
            offsets,
            obj_types=[tile_ty] * n_cores,
            names=[f"{name}{i}" for i in range(n_cores)],
        )

    inA = ObjectFifo(memtile_ty, name="inA")
    inB = ObjectFifo(memtile_ty, name="inB")
    inA_fifos = _split(inA, "memA")
    inB_fifos = _split(inB, "memB")

    outC = ObjectFifo(memtile_ty, name="outC")
    join_offsets = [
        (np.prod(np_ndarray_type_get_shape(memtile_ty)) // n_cores) * i
        for i in range(n_cores)
    ]
    outC_fifos = outC.prod().join(
        join_offsets,
        obj_types=[tile_ty] * n_cores,
        names=[f"memC{i}" for i in range(n_cores)],
    )

    mul_add_fn = _scale_shift_extern(tile_ty)

    rtps = [
        Buffer(
            np.ndarray[(1,), np.dtype[np.int32]],
            name=f"rtp{i}",
            initial_value=np.array([1], dtype=np.int32),
            use_write_rtp=True,
        )
        for i in range(n_cores)
    ]
    barriers = [WorkerRuntimeBarrier() for _ in range(n_cores)]

    def core_fn(of_a, of_b, of_c, mul_add, my_rtp, barrier):
        barrier.wait_for_value(1)
        is_mul = my_rtp[0]
        for _ in range_(tiles_per_core):
            elem_out = of_c.acquire(1)
            elem_in_a = of_a.acquire(1)
            elem_in_b = of_b.acquire(1)
            mul_add(elem_in_a, elem_in_b, elem_out, is_mul)
            of_a.release(1)
            of_b.release(1)
            of_c.release(1)
        barrier.release_with_value(1)

    workers = [
        Worker(
            core_fn,
            fn_args=[
                inA_fifos[i].cons(),
                inB_fifos[i].cons(),
                outC_fifos[i].prod(),
                mul_add_fn,
                rtps[i],
                barriers[i],
            ],
        )
        for i in range(n_cores)
    ]

    def _set_rtps_to(value):
        def _impl(*args):
            for rtp in args:
                rtp[0] = value

        return _impl

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty, tensor_ty) as (A, B, C, D):
        rt.start(*workers)

        # Phase 1: multiply (rtp=1).
        rt.inline_ops(_set_rtps_to(1), rtps)
        for i in range(n_cores):
            rt.set_barrier(barriers[i], 1)
        tg1 = rt.task_group()
        rt.fill(inA.prod(), A, task_group=tg1)
        rt.fill(inB.prod(), B, task_group=tg1)
        rt.drain(outC.cons(), D, wait=True, task_group=tg1)
        rt.finish_task_group(tg1)

        # Phase 2: add (rtp=0).  D = (A*B) feeds back as the lhs.
        rt.inline_ops(_set_rtps_to(0), rtps)
        for i in range(n_cores):
            rt.set_barrier(barriers[i], 1)
        tg2 = rt.task_group()
        rt.fill(inA.prod(), D, task_group=tg2)
        rt.fill(inB.prod(), C, task_group=tg2)
        rt.drain(outC.cons(), D, wait=True, task_group=tg2)
        rt.finish_task_group(tg2)

    return Program(device, rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Scale Shift")
    add_compile_args(p, with_elf=True)
    p.add_argument("-l", "--length", type=int, default=65536, help="elements")
    return p


def _compile_kwargs(opts):
    return dict(size=opts.length)


def _run_and_verify(opts):
    # Constant inputs match the C++ test (4.0, 3.35, 0.77); the two-pass
    # mul-then-add with bf16 intermediate-store makes random inputs flaky
    # to mirror exactly in numpy.
    a_np = np.full((opts.length,), 4.0, dtype=bfloat16)
    b_np = np.full((opts.length,), 3.35, dtype=bfloat16)
    c_np = np.full((opts.length,), 0.77, dtype=bfloat16)
    d_np = np.zeros_like(a_np)

    a_t = iron.tensor(a_np, dtype=bfloat16, device="npu")
    b_t = iron.tensor(b_np, dtype=bfloat16, device="npu")
    c_t = iron.tensor(c_np, dtype=bfloat16, device="npu")
    d_t = iron.tensor(d_np, dtype=bfloat16, device="npu")

    scale_shift(a_t, b_t, c_t, d_t, **_compile_kwargs(opts))

    expected = (
        a_np.astype(np.float32) * b_np.astype(np.float32) + c_np.astype(np.float32)
    ).astype(bfloat16)
    assert_pass(
        d_t.numpy(),
        expected,
        atol=0.002,
        fail_msg="scale_shift output mismatch",
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        scale_shift,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()
