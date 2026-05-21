# aie_trace.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""Vector × scalar with custom hardware event tracing — Iron + @iron.jit.

Same compute as ``basic/vector_scalar_mul`` (a single AIE core scales an
``int32`` vector by a runtime scalar) but with an explicit event list
plumbed through ``rt.enable_trace()``.  The pedagogical point is
showing that the high-level iron Runtime API also accepts the same
``coretile_events`` / ``coremem_events`` / ``memtile_events`` /
``shimtile_events`` knobs as the lower-level ``configure_trace``.

Two invocation modes (mirrors the @iron.jit ports):

  * standalone:   ``python3 aie_trace.py``
  * compile-only: ``... --xclbin-path=PATH --insts-path=PATH``  (Makefile)

The hand-coded ``vector_scalar_mul.cc`` lives next to this file and
gets built into the JIT work_dir via ``ExternalFunction``.
"""

import argparse
import sys
from pathlib import Path

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
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.kernel import ExternalFunction
from aie.utils.hostruntime import set_current_device
from aie.utils.trace.events import (
    CoreEvent,
    MemEvent,
    MemTileEvent,
    MemTilePortEvent,
    PortEvent,
    ShimTileEvent,
    WireBundle,
)


_KERNEL_SRC = str(Path(__file__).parent / "vector_scalar_mul.cc")


def _device_for(dev_str):
    if dev_str == "npu":
        return NPU1Col1()
    if dev_str == "npu2":
        return NPU2Col1()
    raise ValueError(f"[ERROR] Device name {dev_str!r} is unknown")


@iron.jit
def aie_trace(
    A: In,
    F: In,
    C: Out,
    *,
    tensor_size: Compile[int] = 4096,
    tile_size: Compile[int] = 1024,
    trace_size: Compile[int] = 8192,
):
    num_sub_vectors = tensor_size // tile_size

    tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
    scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]
    tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.int32]]

    scale = ExternalFunction(
        "vector_scalar_mul_aie_scalar",
        source_file=_KERNEL_SRC,
        arg_types=[tile_ty, tile_ty, scalar_ty, np.int32],
    )

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
    with rt.sequence(tensor_ty, scalar_ty, tensor_ty) as (a_in, f_in, c_out):
        # Custom per-tile-class event lists, forwarded by iron's Runtime
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
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.fill(of_factor.prod(), f_in)
        rt.drain(of_out.cons(), c_out, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Trace (vector_scalar_mul)")
    p.add_argument("-d", "--dev", type=str, choices=["npu", "npu2"], default="npu")
    p.add_argument("--tensor-size", type=int, default=4096)
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--trace-size", type=int, default=8192)
    p.add_argument("--xclbin-path", type=str, default=None)
    p.add_argument("--insts-path", type=str, default=None)
    return p


def _compile_kwargs(opts):
    return dict(
        tensor_size=opts.tensor_size,
        tile_size=opts.tile_size,
        trace_size=opts.trace_size,
    )


def _compile_only(opts):
    if not opts.insts_path:
        sys.exit("--xclbin-path requires --insts-path (must be set together)")
    set_current_device(_device_for(opts.dev))
    spec = aie_trace.specialize(**_compile_kwargs(opts))
    spec.compile(xclbin_path=opts.xclbin_path, inst_path=opts.insts_path)


def _run_and_verify(opts):
    rng = np.random.default_rng(seed=42)
    a_np = rng.integers(1, 100, size=opts.tensor_size, dtype=np.int32)
    f_np = np.array([3], dtype=np.int32)
    c_np = np.zeros(opts.tensor_size, dtype=np.int32)

    a_t = iron.tensor(a_np, dtype=np.int32, device="npu")
    f_t = iron.tensor(f_np, dtype=np.int32, device="npu")
    c_t = iron.tensor(c_np, dtype=np.int32, device="npu")

    aie_trace(a_t, f_t, c_t, **_compile_kwargs(opts))

    expected = a_np * 3
    if not np.array_equal(c_t.numpy(), expected):
        sys.exit("FAIL! output does not match a * factor")
    print("PASS!")


def main():
    opts = _make_argparser().parse_args()
    if opts.xclbin_path:
        _compile_only(opts)
        return
    _run_and_verify(opts)


if __name__ == "__main__":
    main()
