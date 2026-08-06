# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# The IRON explicit-DMA API exposes DMA constant padding: per-BD pad geometry
# via Bd(pad_dimensions=...) and the per-channel constant pad value via
# DmaChannel(pad_value=...). Check both land on the right ops -- the geometry on
# aie.dma_bd, the value on the aie.dma_start channel op.

import aie.iron as iron
import numpy as np
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir
from aie.iron import Bd, Buffer, DmaChannel, Program, Runtime, TileDma
from aie.iron.device import Tile, from_name


# CHECK: aie.dma_start(MM2S, 0, {{.*}}) {pad_value = 7 : i32}
# CHECK: aie.dma_bd({{.*}} pad [<const_pad_before = 0, const_pad_after = 1>, <const_pad_before = 0, const_pad_after = 0>])
def build_module():
    i8 = np.int8
    mem_ty = np.ndarray[(512,), np.dtype[i8]]

    mem = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    mem_buf = Buffer(type=mem_ty, tile=mem, name="mem_buf")

    mem_dma = TileDma(
        tile=mem,
        channels=[
            DmaChannel(
                direction=DMAChannelDir.MM2S,
                channel=0,
                pad_value=7,
                bds=[
                    Bd(
                        buffer=mem_buf,
                        length=1024,
                        sizes=[1, 512],
                        strides=[512, 1],
                        pad_dimensions=[(0, 1), (0, 0)],
                    ),
                ],
            ),
        ],
    )

    def sequence():
        pass

    rt = Runtime(sequence, [])
    rt.add_tile_dma(mem_dma)
    return Program(iron.get_current_device(), rt).resolve_program()


iron.set_current_device(from_name("npu2", n_cols=1))
print(str(build_module()))
