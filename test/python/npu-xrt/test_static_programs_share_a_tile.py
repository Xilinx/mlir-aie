# test_static_programs_share_a_tile.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""On-device test of two runtimes' TileDma programs on one mem tile.

Each runtime of one configuration loops its input through the mem tile on its
own channel pair, 0 or 1, with a TileDma it registers. The programs merge into
the mem tile's one DMA program, which runs whichever runtime dispatches; a
coordinator runs both.
"""

import aie.iron as iron
import numpy as np
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir
from aie.iron import (
    Acquire,
    Bd,
    Buffer,
    DeviceConfiguration,
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

N = 1024


def loop_through(mem: Tile, channel: int, name: str) -> Runtime:
    """A Runtime whose TileDma loops a tensor through ``mem`` on ``channel``."""
    ty = np.ndarray[(N,), np.dtype[np.int32]]
    shim = Tile(channel, 0, tile_type=AIETileType.ShimNOCTile)
    into = Flow(shim, mem, src_channel=0, dst_channel=channel)
    back = Flow(mem, shim, src_channel=channel, dst_channel=0)
    buffer = Buffer(tile=mem, type=ty, name=f"{name}_buffer")
    free, full = Lock(mem, init=1), Lock(mem, init=0)

    def seq(A, B):
        into.fill(A)
        back.drain(B, wait=True)

    rt = Runtime(seq, [ty, ty], name=name)
    rt.add_flow(into)
    rt.add_flow(back)
    rt.add_tile_dma(
        TileDma(
            mem,
            [
                DmaChannel(
                    DMAChannelDir.S2MM,
                    channel,
                    [Bd(buffer, acquires=[Acquire(free)], releases=[Release(full)])],
                ),
                DmaChannel(
                    DMAChannelDir.MM2S,
                    channel,
                    [Bd(buffer, acquires=[Acquire(full)], releases=[Release(free)])],
                ),
            ],
        )
    )
    return rt


# The coordinator configures the device, which takes a full ELF.
@iron.jit(full_elf=True)
def both(a: In, b: Out, c: In, d: Out):
    ty = np.ndarray[(N,), np.dtype[np.int32]]
    mem = Tile(0, 1, tile_type=AIETileType.MemTile)
    first = loop_through(mem, 0, "first")
    second = loop_through(mem, 1, "second")

    def coordinate(A, B, C, D):
        with pack.configure():
            first.call(A, B)
            second.call(C, D)

    coordinator = Runtime(
        coordinate, [ty] * 4, name="coordinate", implicit_configure=False
    )
    pack = DeviceConfiguration(
        "pack", iron.get_current_device(), runtimes=[first, second, coordinator]
    )
    return Program.compose([pack], entry=coordinator).resolve_program()


def test_two_runtimes_programs_run_on_one_mem_tile():
    rng = np.random.default_rng(8)
    first = rng.integers(-(2**31), 2**31 - 1, N, dtype=np.int32)
    second = rng.integers(-(2**31), 2**31 - 1, N, dtype=np.int32)
    a = iron.tensor(first, dtype=np.int32, device="npu")
    c = iron.tensor(second, dtype=np.int32, device="npu")
    b = iron.zeros(N, dtype=np.int32, device="npu")
    d = iron.zeros(N, dtype=np.int32, device="npu")
    both(a, b, c, d)
    b.to("cpu")
    d.to("cpu")
    np.testing.assert_array_equal(b.numpy(), first)
    np.testing.assert_array_equal(d.numpy(), second)
