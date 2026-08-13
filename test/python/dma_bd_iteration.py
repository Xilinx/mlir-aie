# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""Test that Bd.iteration forwards the BD iteration state to the underlying
aie.dma_bd op as an #aie.bd_iteration attribute."""

import numpy as np

from aie.iron import Bd, BdIteration, Buffer, DmaChannel, Program, Runtime, TileDma
from aie.iron.device import NPU2Col1, Tile
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir


def emit_iteration_bd():
    n = 256
    vector_ty = np.ndarray[(n,), np.dtype[np.int32]]

    compute_tile = Tile(col=0, row=2, tile_type=AIETileType.CoreTile)
    buf = Buffer(tile=compute_tile, type=vector_ty, name="iteration_buf")

    tile_dma = TileDma(
        tile=compute_tile,
        channels=[
            DmaChannel(
                direction=DMAChannelDir.MM2S,
                channel=0,
                bds=[
                    Bd(
                        buffer=buf,
                        offset=0,
                        length=n,
                        sizes=[16, 16],
                        strides=[16, 1],
                        iteration=BdIteration(size=4, stride=16, current=2),
                        next="self",
                    ),
                ],
            ),
        ],
    )

    def sequence(_):
        pass

    rt = Runtime(sequence, [vector_ty])
    rt.add_tile_dma(tile_dma)

    return Program(NPU2Col1(), rt).resolve_program()


# CHECK: aie.dma_bd({{.*}} : memref<256xi32> len = {{.*}} sizes = [16, 16] strides = [16, 1])
# CHECK-SAME: iteration = #aie.bd_iteration<size = 4, stride = 16, current = 2>
print(emit_iteration_bd())
