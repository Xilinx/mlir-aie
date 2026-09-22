# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | aie-opt --pass-pipeline='any(aie.device(aie-place-tiles,aie-objectFifo-stateful-transform))' | FileCheck %s

"""Test pinning every hardware DMA channel an ObjectFifo pair uses.

prod()/cons() take a channel, but forward() builds the forwarded fifo's
producer handle itself -- so without a channel of its own that one endpoint is
left to the compiler while the other three are pinned. This checks all four
land where they were asked to.

Also checks that forwarding through a compute tile works from coordinates
alone: forward() defaults to a mem tile, and stamping that over an already
placed tile contradicts what the Device infers from its row.
"""

import aie.iron as iron
import numpy as np
from aie.iron import ObjectFifo, Program, Runtime
from aie.iron.device import NPU2Col1, Tile

chunk = np.ndarray[(512,), np.dtype[np.int8]]


def pinned():
    tile = Tile(0, 5)  # a compute tile, by coordinates only
    shim = Tile(0, 0)

    into = ObjectFifo(chunk, depth=2, name="into")
    out = into.cons(channel=1).forward(tile=tile, channel=1, name="out")

    def sequence(a, c, fill_from, drain_to):
        fill_from.fill(a)
        drain_to.drain(c, wait=True)

    rt = Runtime(
        sequence,
        [
            np.ndarray[(4096,), np.dtype[np.int8]],
            np.ndarray[(4096,), np.dtype[np.int8]],
            into.prod(tile=shim, channel=1),
            out.cons(tile=shim, channel=1),
        ],
    )
    return Program(NPU2Col1(), rt).resolve_program()


iron.set_current_device(NPU2Col1())
print(pinned())

# The staging tile is the compute tile named, not a mem tile.
# CHECK: %tile_0_5 = aie.tile(0, 5)

# All four channels are the ones asked for -- 1, not the 0 auto-assignment
# would have picked.
# CHECK-DAG: aie.shim_dma_allocation @into_shim_alloc(%{{.*}}, MM2S, 1)
# CHECK-DAG: aie.shim_dma_allocation @out_shim_alloc(%{{.*}}, S2MM, 1)
# CHECK:     aie.mem(%tile_0_5)
# CHECK-DAG:   aie.dma_start(S2MM, 1
# CHECK-DAG:   aie.dma_start(MM2S, 1
