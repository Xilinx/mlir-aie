# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""Test that Bd.sizes/strides forward a multi-dimensional (strided) access
pattern to the underlying aie.dma_bd op, emitting the sizes/strides clause
in the lowered MLIR."""

import numpy as np

from aie.iron import Bd, Buffer, DmaChannel, Program, Runtime, TileDma
from aie.iron.device import NPU2Col1, Tile
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir


def emit_strided_bd():
    n = 256
    vector_ty = np.ndarray[(n,), np.dtype[np.int32]]

    compute_tile = Tile(col=0, row=2, tile_type=AIETileType.CoreTile)
    buf = Buffer(tile=compute_tile, type=vector_ty, name="strided_buf")

    # A 2-D strided access pattern: outer 16x(stride 16), inner 16x(stride 1).
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
print(emit_strided_bd())
