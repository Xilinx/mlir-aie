# aie_trace.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""Vector × scalar with custom hardware event tracing — IRON + @iron.jit.

Same compute as ``basic/vector_scalar_mul`` (a single AIE core scales an
``int32`` vector by a runtime scalar) but with an explicit event list
plumbed through ``rt.enable_trace()``.  The pedagogical point is
showing that the high-level IRON Runtime API also accepts the same
``coretile_events`` / ``coremem_events`` / ``memtile_events`` /
``shimtile_events`` parameters as the lower-level ``configure_trace``.

Two invocation modes:

  * standalone:   ``python3 aie_trace.py``
  * compile-only: ``... --xclbin-path=PATH --insts-path=PATH``  (Makefile)
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import (
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    kernels,
)
from aie.iron.controlflow import range_
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass
from aie.utils.trace.events import (
    CoreEvent,
    MemEvent,
    MemTileEvent,
    MemTilePortEvent,
    PortEvent,
    ShimTileEvent,
    WireBundle,
)


@iron.jit
def aie_trace(
    A: In,
    F: In,
    C: Out,
    *,
    tensor_size: CompileTime[int] = 4096,
    tile_size: CompileTime[int] = 1024,
    trace_size: CompileTime[int] = 8192,
):
    num_sub_vectors = tensor_size // tile_size

    tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
    scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]
    tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.int32]]

    scale = kernels.scale(tile_size=tile_size, dtype=np.int32, vectorized=False)

    of_in = ObjectFifo(tile_ty, name="in")
    of_factor = ObjectFifo(scalar_ty, name="infactor")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_fn(of_in, of_factor, of_out, scale):
        elem_factor = of_factor.acquire(1)
        for _ in range_(num_sub_vectors):
            elem_out = of_out.acquire(1)
            elem_in = of_in.acquire(1)
            scale(elem_in, elem_out, elem_factor, tile_size)
            of_in.release(1)
            of_out.release(1)
        of_factor.release(1)

    worker = Worker(
        core_fn,
        fn_args=[of_in.cons(), of_factor.cons(), of_out.prod(), scale],
        trace=1,
    )

    rt = Runtime()

    def sequence(a_in, f_in, c_out):
        # Custom per-tile-class event lists, forwarded by IRON's Runtime
        # to the same configure_trace() the dialect-level example used.
        rt.enable_trace(
            trace_size=trace_size,
            workers=[worker],
            coretile_events=[
                CoreEvent.INSTR_EVENT_0,
                CoreEvent.INSTR_EVENT_1,
                CoreEvent.INSTR_VECTOR,
                CoreEvent.MEMORY_STALL,
                CoreEvent.STREAM_STALL,
                CoreEvent.LOCK_STALL,
                PortEvent(CoreEvent.PORT_RUNNING_0, WireBundle.DMA, 0, True),
                PortEvent(CoreEvent.PORT_RUNNING_1, WireBundle.DMA, 0, False),
            ],
            coremem_events=[
                MemEvent.DMA_S2MM_0_START_TASK,
                MemEvent.DMA_S2MM_1_START_TASK,
                MemEvent.DMA_MM2S_0_START_TASK,
                MemEvent.DMA_S2MM_0_FINISHED_TASK,
                MemEvent.DMA_S2MM_1_FINISHED_TASK,
                MemEvent.DMA_MM2S_0_FINISHED_TASK,
                MemEvent.DMA_S2MM_0_STREAM_STARVATION,
                MemEvent.DMA_S2MM_1_STREAM_STARVATION,
            ],
            memtile_events=[
                MemTilePortEvent(MemTileEvent.PORT_RUNNING_0, WireBundle.DMA, 0, False),
                MemTilePortEvent(MemTileEvent.PORT_RUNNING_1, WireBundle.DMA, 1, False),
                MemTilePortEvent(MemTileEvent.PORT_RUNNING_2, WireBundle.DMA, 0, True),
                MemTilePortEvent(MemTileEvent.PORT_RUNNING_3, WireBundle.DMA, 1, True),
                MemTilePortEvent(MemTileEvent.PORT_RUNNING_4, WireBundle.DMA, 2, True),
                MemTilePortEvent(MemTileEvent.PORT_RUNNING_5, WireBundle.DMA, 3, True),
                MemTilePortEvent(MemTileEvent.PORT_RUNNING_6, WireBundle.DMA, 4, True),
                MemTilePortEvent(MemTileEvent.PORT_RUNNING_7, WireBundle.DMA, 5, True),
            ],
            shimtile_events=[
                ShimTileEvent.DMA_S2MM_0_START_TASK,
                ShimTileEvent.DMA_S2MM_1_START_TASK,
                ShimTileEvent.DMA_MM2S_0_START_TASK,
                ShimTileEvent.DMA_S2MM_0_FINISHED_TASK,
                ShimTileEvent.DMA_S2MM_1_FINISHED_TASK,
                ShimTileEvent.DMA_MM2S_0_FINISHED_TASK,
                ShimTileEvent.DMA_S2MM_0_STREAM_STARVATION,
                ShimTileEvent.DMA_S2MM_1_STREAM_STARVATION,
            ],
        )
        of_in.prod().fill(a_in)
        of_factor.prod().fill(f_in)
        of_out.cons().drain(c_out, wait=True)

    rt.sequence(sequence, [tensor_ty, scalar_ty, tensor_ty])

    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Trace (vector_scalar_mul)")
    add_compile_args(p)
    p.add_argument("--tensor-size", type=int, default=4096)
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--trace-size", type=int, default=8192)
    return p


def _compile_kwargs(opts):
    return dict(
        tensor_size=opts.tensor_size,
        tile_size=opts.tile_size,
        trace_size=opts.trace_size,
    )


def _run_and_verify(opts):
    rng = np.random.default_rng(seed=42)
    a_np = rng.integers(1, 100, size=opts.tensor_size, dtype=np.int32)
    a_t = iron.tensor(a_np, dtype=np.int32, device="npu")
    f_t = iron.full((1,), 3, dtype=np.int32, device="npu")
    c_t = iron.zeros(opts.tensor_size, dtype=np.int32, device="npu")

    aie_trace(a_t, f_t, c_t, **_compile_kwargs(opts))

    expected = a_np * 3
    assert_pass(c_t.numpy(), expected, fail_msg="output does not match a * factor")


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        aie_trace,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
    )


if __name__ == "__main__":
    main()
