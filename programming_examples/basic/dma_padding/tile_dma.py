# dma_padding/tile_dma.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""DMA constant padding via the explicit TileDma interface.

One entrypoint exposes ``pad_value`` here: a hand-placed memtile ``TileDma``
whose MM2S ``DmaChannel(pad_value=...)`` sets the per-channel fill and whose
``Bd(pad_dimensions=...)`` sets the per-BD geometry. Stages a transfer
shim -> memtile -> shim. See harness.py for the run/verify sweep and pad cases.
"""

import aie.iron as iron
import numpy as np
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir
from aie.dialects.aie import EndOp  # pyright: ignore[reportAttributeAccessIssue]
from aie.dialects.aiex import (
    bds,
    dma_await_task,
    dma_configure_task,
    dma_free_task,
    dma_start_task,
    shim_dma_bd,
)
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
)
from aie.iron.device import Tile
from harness import PAD_AFTER, PAD_BEFORE, REAL, REGION, main


def _dma_channel(elem_dtype):
    @iron.jit
    def dma_channel(a_in: In, c_out: Out, *, pad_value: CompileTime[int] = 0):
        mem_ty = np.ndarray[(REAL,), np.dtype[elem_dtype]]
        out_ty = np.ndarray[(REGION,), np.dtype[elem_dtype]]

        shim = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
        mem = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
        buf = Buffer(type=mem_ty, tile=mem, name="mem_buf")
        prod, cons = Lock(tile=mem, init=1, name="p"), Lock(tile=mem, init=0, name="c")

        # Bd carries the per-BD pad geometry; DmaChannel the per-channel value.
        mem_dma = TileDma(
            tile=mem,
            channels=[
                DmaChannel(
                    direction=DMAChannelDir.S2MM,
                    channel=0,
                    bds=[
                        Bd(
                            buffer=buf,
                            length=REAL,
                            acquires=[Acquire(prod)],
                            releases=[Release(cons)],
                        )
                    ],
                ),
                DmaChannel(
                    direction=DMAChannelDir.MM2S,
                    channel=0,
                    pad_value=pad_value,
                    bds=[
                        Bd(
                            buffer=buf,
                            length=REGION,
                            sizes=[REAL],
                            strides=[1],
                            pad_dimensions=[(PAD_BEFORE, PAD_AFTER)],
                            acquires=[Acquire(cons)],
                            releases=[Release(prod)],
                        )
                    ],
                ),
            ],
        )

        def sequence(a, c):
            in_task = dma_configure_task(shim.op, DMAChannelDir.MM2S, 0)
            with bds(in_task) as bd:
                with bd[0]:
                    shim_dma_bd(
                        a.op, offset=0, sizes=[1, 1, 1, REAL], strides=[0, 0, 0, 1]
                    )
                    EndOp()
            out_task = dma_configure_task(
                shim.op, DMAChannelDir.S2MM, 0, issue_token=True
            )
            with bds(out_task) as bd:
                with bd[0]:
                    shim_dma_bd(
                        c.op, offset=0, sizes=[1, 1, 1, REGION], strides=[0, 0, 0, 1]
                    )
                    EndOp()
            dma_start_task(in_task, out_task)
            dma_await_task(out_task)
            dma_free_task(in_task)

        rt = Runtime(sequence, [mem_ty, out_ty])
        rt.add_flow(Flow(src=shim, dst=mem, src_channel=0, dst_channel=0))
        rt.add_flow(Flow(src=mem, dst=shim, src_channel=0, dst_channel=0))
        rt.add_lock(prod)
        rt.add_lock(cons)
        rt.add_tile_dma(mem_dma)
        return Program(iron.get_current_device(), rt).resolve_program()

    return dma_channel


if __name__ == "__main__":
    main({"dma_channel": _dma_channel})
