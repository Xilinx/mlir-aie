# test_tile_dma_task_token.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""On-device test of awaiting a mem tile task's completion token.

The token leaves the mem tile through its TileControl port, so the host only
sees it if the compiler routes that port to the shim. The sequence awaits the
S2MM task before it starts the MM2S one, so the token alone orders the two
halves: no locks. Both tasks run on Flow endpoints whose channels the compiler
assigns.
"""

import aie.iron as iron
import numpy as np
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir
from aie.iron import Buffer, Flow, In, Out, Program, Runtime, tile_dma_task
from aie.iron.device import Tile

N = 1024


@iron.jit
def round_trip(a: In, c: Out):
    ty = np.ndarray[(N,), np.dtype[np.int32]]
    shim = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
    mem = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    resident = Buffer(type=ty, tile=mem, name="resident")
    into = Flow(shim, mem)
    out = Flow(mem, shim)

    def seq(A, C):
        into.fill(A)
        tile_dma_task(
            mem, DMAChannelDir.S2MM, into.endpoint(mem), resident, wait=True
        ).await_()
        tile_dma_task(mem, DMAChannelDir.MM2S, out.endpoint(mem), resident)
        out.drain(C, wait=True)

    rt = Runtime(seq, [ty, ty])
    rt.add_flow(into)
    rt.add_flow(out)
    rt.add_buffer(resident)
    return Program(iron.get_current_device(), rt).resolve_program()


def test_tile_dma_task_token():
    a = iron.tensor(
        np.random.default_rng(0).integers(0, 2**16, size=(N,), dtype=np.int32),
        dtype=np.int32,
        device="npu",
    )
    c = iron.zeros((N,), dtype=np.int32, device="npu")
    round_trip(a, c)
    np.testing.assert_array_equal(c.numpy(), a.numpy())
