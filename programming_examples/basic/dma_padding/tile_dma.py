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
from aie.dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    AIETileType,
    DMAChannelDir,
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
        into, out = Flow(shim, mem), Flow(mem, shim)

        # Bd carries the per-BD pad geometry; DmaChannel the per-channel value.
        mem_dma = TileDma(
            tile=mem,
            channels=[
                DmaChannel(
                    direction=DMAChannelDir.S2MM,
                    channel=into.endpoint(mem),
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
                    channel=out.endpoint(mem),
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
            into.fill(a)
            out.drain(c, wait=True)

        rt = Runtime(sequence, [mem_ty, out_ty])
        rt.add_flow(into)
        rt.add_flow(out)
        rt.add_lock(prod)
        rt.add_lock(cons)
        rt.add_tile_dma(mem_dma)
        return Program(iron.get_current_device(), rt).resolve_program()

    return dma_channel


if __name__ == "__main__":
    main({"dma_channel": _dma_channel})
