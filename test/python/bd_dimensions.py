# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""Test that a Bd's tap forwards a multi-dimensional (strided) access
pattern to the underlying aie.dma_bd op, emitting the sizes/strides clause
in the lowered MLIR."""

import numpy as np

from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Bd, Buffer, DmaChannel, Program, Runtime, TileDma
from aie.iron.device import NPU2Col1, Tile
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir


def emit_strided_bd():
    n = 256
    vector_ty = np.ndarray[(n,), np.dtype[np.int32]]

    compute_tile = Tile(col=0, row=2, tile_type=AIETileType.CoreTile)
    buf = Buffer(tile=compute_tile, type=vector_ty, name="strided_buf")

    # A 2-D strided access pattern: the column-major walk of a 16x16 tile.
    tile_dma = TileDma(
        tile=compute_tile,
        channels=[
            DmaChannel(
                direction=DMAChannelDir.MM2S,
                channel=0,
                bds=[
                    Bd(buffer=buf, tap=TensorAccessPattern.full((16, 16)).T),
                ],
            ),
        ],
    )

    def sequence(_):
        pass

    rt = Runtime(sequence, [vector_ty])
    rt.add_tile_dma(tile_dma)

    return Program(NPU2Col1(), rt).resolve_program()


# CHECK: aie.dma_bd({{.*}} : memref<256xi32> len = 256 sizes = [16, 16] strides = [1, 16])
print(emit_strided_bd())
