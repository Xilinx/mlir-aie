# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""Test DmaChannel.loop, which decides where the last BD in a chain points.

Looping (the default) chains it back to the head, so the chain is endless and
runs for as long as its locks allow. Not looping points it at the region's
aie.end, which ends the chain -- and only a chain that ends is a task the
channel can re-run, so that is what makes repeat_count mean anything."""

import numpy as np

from aie.iron import Bd, Buffer, DmaChannel, Program, Runtime, TileDma
from aie.iron.device import NPU2Col1, Tile
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir


def emit_chains():
    n = 256
    vector_ty = np.ndarray[(n,), np.dtype[np.int32]]

    compute_tile = Tile(col=0, row=2, tile_type=AIETileType.CoreTile)
    looping_buf = Buffer(tile=compute_tile, type=vector_ty, name="looping_buf")
    ending_buf = Buffer(tile=compute_tile, type=vector_ty, name="ending_buf")

    tile_dma = TileDma(
        tile=compute_tile,
        channels=[
            DmaChannel(
                direction=DMAChannelDir.MM2S,
                channel=0,
                bds=[Bd(buffer=looping_buf, length=n)],
            ),
            DmaChannel(
                direction=DMAChannelDir.MM2S,
                channel=1,
                loop=False,
                repeat_count=3,
                bds=[Bd(buffer=ending_buf, length=n)],
            ),
        ],
    )

    def sequence(_):
        pass

    rt = Runtime(sequence, [vector_ty])
    rt.add_tile_dma(tile_dma)

    return Program(NPU2Col1(), rt).resolve_program()


# The looping channel's BD points back at its own block.
# CHECK:      aie.dma_start(MM2S, 0, ^[[HEAD:.*]], ^{{.*}})
# CHECK:      ^[[HEAD]]:
# CHECK:        aie.dma_bd(%looping_buf
# CHECK:        aie.next_bd ^[[HEAD]]

# The non-looping channel's BD points at the aie.end block instead, so the
# chain ends and repeat_count re-runs it.
# CHECK:      aie.dma_start(MM2S, 1, ^{{.*}}, ^{{.*}}, repeat_count = 3)
# CHECK:        aie.dma_bd(%ending_buf
# CHECK:        aie.next_bd ^[[END:.*]]
# CHECK:      ^[[END]]:
# CHECK-NEXT:   aie.end
print(emit_chains())
