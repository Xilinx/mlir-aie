# test_tile_dma_pad_value.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""On-device test of DMA constant padding through the IRON explicit-DMA API.

A hand-placed memtile TileDma stages a 13-element int32 transfer and its MM2S
channel pads it up to 16 (2 before, 1 after), filling the padded region with
DmaChannel(pad_value=42) and the geometry via Bd(pad_dimensions=...). Pure DMA
passthrough (no core), so the read-back directly exposes the pad fill.
"""

import aie.iron as iron
import numpy as np
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir
from aie.dialects.aie import EndOp
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

REAL = 13
REGION = 16
PAD_BEFORE = 2
PAD_AFTER = 1
PAD_VALUE = 42


@iron.jit
def tile_dma_pad(a: In, c: Out):
    mem_ty = np.ndarray[(REAL,), np.dtype[np.int32]]
    shim = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
    mem = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    mem_buf = Buffer(type=mem_ty, tile=mem, name="mem_buf")
    p, cl = Lock(tile=mem, init=1, name="p"), Lock(tile=mem, init=0, name="c")

    # Bd carries the per-BD pad geometry; DmaChannel carries the per-channel value.
    mem_dma = TileDma(
        tile=mem,
        channels=[
            DmaChannel(
                direction=DMAChannelDir.S2MM,
                channel=0,
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
                channel=0,
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
    flow_in = Flow(src=shim, dst=mem, src_channel=0, dst_channel=0)
    flow_out = Flow(src=mem, dst=shim, src_channel=0, dst_channel=0)

    def seq(A, C):
        a_task = dma_configure_task(shim.op, DMAChannelDir.MM2S, 0)
        with bds(a_task) as bd:
            with bd[0]:
                shim_dma_bd(A.op, offset=0, sizes=[1, 1, 1, REAL], strides=[0, 0, 0, 1])
                EndOp()
        out_task = dma_configure_task(shim.op, DMAChannelDir.S2MM, 0, issue_token=True)
        with bds(out_task) as bd:
            with bd[0]:
                shim_dma_bd(
                    C.op, offset=0, sizes=[1, 1, 1, REGION], strides=[0, 0, 0, 1]
                )
                EndOp()
        dma_start_task(a_task, out_task)
        dma_await_task(out_task)
        dma_free_task(a_task)

    rt = Runtime(
        seq,
        [
            np.ndarray[(REAL,), np.dtype[np.int32]],
            np.ndarray[(REGION,), np.dtype[np.int32]],
        ],
    )
    for f in (flow_in, flow_out):
        rt.add_flow(f)
    for lk in (p, cl):
        rt.add_lock(lk)
    rt.add_tile_dma(mem_dma)
    return Program(iron.get_current_device(), rt).resolve_program()


def test_tile_dma_pad_value():
    a = iron.arange(REAL, dtype=np.int32)  # 0..12
    c = iron.zeros(REGION, dtype=np.int32, device="npu")
    tile_dma_pad(a, c)
    c.to("cpu")

    expected = np.array(
        [PAD_VALUE] * PAD_BEFORE + list(range(REAL)) + [PAD_VALUE] * PAD_AFTER,
        dtype=np.int32,
    )
    np.testing.assert_array_equal(c.numpy(), expected)
