# test_tile_dma_pad_value.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""On-device test of DMA constant padding through the IRON explicit-DMA API.

A hand-placed memtile TileDma stages a 8-element int8 transfer and its MM2S
channel pads it up to 16 (4 before, 4 after), filling the padded region with
DmaChannel(pad_value=42) and the geometry via Bd(pad_dimensions=...). Pure DMA
passthrough (no core), so the read-back directly exposes the pad fill.
"""

import aie.iron as iron
import numpy as np
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir
from aie.iron import (
    Acquire,
    Bd,
    Buffer,
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

REAL = 8
REGION = 16
PAD_BEFORE = 4
PAD_AFTER = 4
PAD_VALUE = 42


@iron.jit
def tile_dma_pad(a: In, c: Out):
    mem_ty = np.ndarray[(REAL,), np.dtype[np.int8]]
    shim = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
    mem = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    mem_buf = Buffer(type=mem_ty, tile=mem, name="mem_buf")
    p, cl = Lock(tile=mem, init=1, name="p"), Lock(tile=mem, init=0, name="c")
    flow_in, flow_out = Flow(shim, mem), Flow(mem, shim)

    # Bd carries the per-BD pad geometry; DmaChannel carries the per-channel value.
    mem_dma = TileDma(
        tile=mem,
        channels=[
            DmaChannel(
                direction=DMAChannelDir.S2MM,
                channel=flow_in.endpoint(mem),
                bds=[
                    Bd(
                        buffer=mem_buf,
                        length=REAL,
                        acquires=[Acquire(p)],
                        releases=[Release(cl)],
                    )
                ],
            ),
            DmaChannel(
                direction=DMAChannelDir.MM2S,
                channel=flow_out.endpoint(mem),
                pad_value=PAD_VALUE,
                bds=[
                    Bd(
                        buffer=mem_buf,
                        length=REGION,
                        sizes=[REAL],
                        strides=[1],
                        pad_dimensions=[(PAD_BEFORE, PAD_AFTER)],
                        acquires=[Acquire(cl)],
                        releases=[Release(p)],
                    )
                ],
            ),
        ],
    )

    def seq(A, C):
        flow_in.fill(A)
        flow_out.drain(C, wait=True)

    rt = Runtime(
        seq,
        [
            np.ndarray[(REAL,), np.dtype[np.int8]],
            np.ndarray[(REGION,), np.dtype[np.int8]],
        ],
    )
    for f in (flow_in, flow_out):
        rt.add_flow(f)
    for lk in (p, cl):
        rt.add_lock(lk)
    rt.add_tile_dma(mem_dma)
    return Program(iron.get_current_device(), rt).resolve_program()


def test_tile_dma_pad_value():
    a = iron.arange(REAL, dtype=np.int8)  # 0..12
    c = iron.zeros(REGION, dtype=np.int8, device="npu")
    tile_dma_pad(a, c)
    c.to("cpu")

    expected = np.array(
        [PAD_VALUE] * PAD_BEFORE + list(range(REAL)) + [PAD_VALUE] * PAD_AFTER,
        dtype=np.int8,
    )
    np.testing.assert_array_equal(c.numpy(), expected)
