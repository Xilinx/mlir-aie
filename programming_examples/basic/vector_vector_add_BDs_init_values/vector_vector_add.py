# vector_vector_add_BDs_init_values/vector_vector_add.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Vector + vector at the iron BD level, with one operand baked into L1.

Two pedagogical points, both visible in the design body:

  1. **BD-level data movement.**  Instead of letting :class:`ObjectFifo`
     manage routing, DMA, and lock handshakes, this design hand-wires
     them via the iron BD-level primitives :class:`Flow`, :class:`Lock`,
     :class:`TileDma`, :class:`DmaChannel`, :class:`Bd`, :class:`Acquire`,
     :class:`Release`.  The core body explicitly acquires / releases the
     producer / consumer locks that synchronise with these BDs.

  2. **``Buffer(initial_value=array)``.**  The second operand lives in a
     :class:`PreInitializedConstantBuffer` (a :class:`Buffer` subclass)
     whose contents are written into L1 at design startup, so no shim
     DMA is needed for it.  Compare with
     ``programming_examples/basic/custom_dma/`` for a richer user-side
     :class:`Resolvable` example.

Two invocation modes:

  * compile-only: ``... --xclbin-path=PATH --insts-path=PATH``  (NPU Makefile)
  * emit-MLIR:    ``... -d xcvc1902 --emit-mlir``               (vck5000)
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import (
    Acquire,
    Bd,
    Buffer,
    CompileTime,
    DmaChannel,
    Flow,
    In,
    Lock,
    Out,
    Program,
    Release,
    Runtime,
    TileDma,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.utils.hostruntime.argparse import device_from_args
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir, WireBundle
from aie.dialects.aiex import (
    dma_await_task,
    dma_free_task,
    dma_start_task,
    shim_dma_single_bd_task,
)
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli


class PreInitializedConstantBuffer(Buffer):
    """L1 buffer whose contents are baked into the design at startup.

    A thin :class:`Buffer` subclass demonstrating the
    ``initial_value=`` mechanism as a named, reusable component.
    """

    def __init__(self, value: np.ndarray, name: str | None = None):
        super().__init__(
            type=np.ndarray[value.shape, np.dtype[value.dtype]],
            name=name,
            initial_value=value,
        )


@iron.jit
def vector_vector_add(
    A: In,
    C: Out,
    *,
    col: CompileTime[int] = 0,
):
    dev = iron.get_current_device()
    N = 256
    n = 16
    N_div_n = N // n

    tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]
    tile_ty = np.ndarray[(n,), np.dtype[np.int32]]

    shim_tile = Tile(col=col, row=0, tile_type=AIETileType.ShimNOCTile)
    compute_tile = Tile(col=col, row=2, tile_type=AIETileType.CoreTile)

    # All buffers and locks live on the compute tile.  Buffers are auto-
    # placed by Worker when passed in fn_args; we just need to give them a
    # type + name.
    in1_buff = Buffer(type=tile_ty, name="in1_cons_buff_0")
    in2_buff = PreInitializedConstantBuffer(
        np.arange(N, dtype=np.int32), name="in2_cons_buff_0"
    )
    out_buff = Buffer(type=tile_ty, name="out_buff_0")

    in1_prod_lock = Lock(
        tile=compute_tile, lock_id=0, init=1, name="in1_cons_prod_lock"
    )
    in1_cons_lock = Lock(
        tile=compute_tile, lock_id=1, init=0, name="in1_cons_cons_lock"
    )
    in2_prod_lock = Lock(
        tile=compute_tile, lock_id=2, init=0, name="in2_cons_prod_lock"
    )
    in2_cons_lock = Lock(
        tile=compute_tile, lock_id=3, init=1, name="in2_cons_cons_lock"
    )
    out_prod_lock = Lock(tile=compute_tile, lock_id=4, init=1, name="out_prod_lock")
    out_cons_lock = Lock(tile=compute_tile, lock_id=5, init=0, name="out_cons_lock")

    # Explicit routes: shim → compute → shim, plus the shim_dma_allocation
    # symbols the runtime sequence references by name.
    in_flow = Flow(
        src=shim_tile,
        dst=compute_tile,
        src_port=WireBundle.DMA,
        src_channel=0,
        dst_port=WireBundle.DMA,
        dst_channel=0,
        shim_symbol="of_in1",
    )
    out_flow = Flow(
        src=compute_tile,
        dst=shim_tile,
        src_port=WireBundle.DMA,
        src_channel=0,
        dst_port=WireBundle.DMA,
        dst_channel=0,
        shim_symbol="of_out",
    )

    # Per-tile DMA program: S2MM ch0 fills in1_buff; MM2S ch0 drains out_buff.
    compute_dma = TileDma(
        tile=compute_tile,
        channels=[
            DmaChannel(
                direction=DMAChannelDir.S2MM,
                channel=0,
                bds=[
                    Bd(
                        buffer=in1_buff,
                        acquires=[Acquire(in1_prod_lock)],
                        releases=[Release(in1_cons_lock)],
                        next="self",
                    ),
                ],
            ),
            DmaChannel(
                direction=DMAChannelDir.MM2S,
                channel=0,
                bds=[
                    Bd(
                        buffer=out_buff,
                        acquires=[Acquire(out_cons_lock)],
                        releases=[Release(out_prod_lock)],
                        next="self",
                    ),
                ],
            ),
        ],
    )

    def core_body(in1, in2, out, in1_p, in1_c, in2_p, in2_c, out_p, out_c):
        # Worker wraps this in `for _ in range_(sys.maxsize)` by default
        # (while_true=True).  Locks + buffers are shared by reference with
        # compute_dma above.
        in2_c.acquire()
        for j in range_(N_div_n):
            in1_c.acquire()
            out_p.acquire()
            for i in range_(n):
                out[i] = in2[j * N_div_n + i] + in1[i]
            in1_p.release()
            out_c.release()
        in2_p.release()

    worker = Worker(
        core_body,
        fn_args=[
            in1_buff,
            in2_buff,
            out_buff,
            in1_prod_lock,
            in1_cons_lock,
            in2_prod_lock,
            in2_cons_lock,
            out_prod_lock,
            out_cons_lock,
        ],
        tile=compute_tile,
    )

    def emit_seq(A_data, C_data):
        in1_task = shim_dma_single_bd_task("of_in1", A_data.op, sizes=[1, 1, 1, N])
        out_task = shim_dma_single_bd_task(
            "of_out", C_data.op, sizes=[1, 1, 1, N], issue_token=True
        )
        dma_start_task(in1_task, out_task)
        dma_await_task(out_task)
        dma_free_task(in1_task)

    rt = Runtime()
    rt.add_flow(in_flow)
    rt.add_flow(out_flow)
    for lk in (
        in1_prod_lock,
        in1_cons_lock,
        in2_prod_lock,
        in2_cons_lock,
        out_prod_lock,
        out_cons_lock,
    ):
        rt.add_lock(lk)
    rt.add_tile_dma(compute_dma)

    def sequence(A, C):
        rt.inline_ops(emit_seq, [A, C])

    rt.sequence(sequence, [tensor_ty, tensor_ty])

    return Program(dev, rt, workers=[worker]).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Vector Vector Add (BDs init values)")
    add_compile_args(p, dev_choices=("npu", "npu2", "xcvc1902"), with_emit_mlir=True)
    p.add_argument("-c", "--col", type=int, default=0)
    return p


def _compile_kwargs(opts):
    return dict(col=opts.col)


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        vector_vector_add,
        opts,
        compile_kwargs=_compile_kwargs,
        device=device_from_args,
    )


if __name__ == "__main__":
    main()
