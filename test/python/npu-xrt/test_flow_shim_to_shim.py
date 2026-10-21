# test_flow_shim_to_shim.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""On-device test of a route from one shim tile to another.

fill() drives the source shim's MM2S and drain() the destination shim's S2MM,
so data crosses the shim row without touching the array: every other element
of the input leaves column 0 and lands in column 1's output, as a gather's
feed and drain shims pass it.
"""

import aie.iron as iron
import numpy as np
import pytest
from aie.dialects._aie_enum_gen import AIETileType
from aie.iron import CompileTime, Flow, In, Out, PacketFlow, Program, Runtime
from aie.iron.device import Tile

N = 1024


@iron.jit
def shim_to_shim(a: In, c: Out, *, route: CompileTime[str] = "flow"):
    in_ty = np.ndarray[(N,), np.dtype[np.int32]]
    out_ty = np.ndarray[(N // 2,), np.dtype[np.int32]]
    feed = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
    drain = Tile(col=1, row=0, tile_type=AIETileType.ShimNOCTile)
    if route == "flow":
        link = Flow(feed, drain, src_channel=1, dst_channel=0)
    elif route == "routed":
        link = Flow(feed, drain)
    else:
        link = PacketFlow(3, feed, drain, src_channel=1)

    def seq(A, C):
        link.fill(A, tap=A[::2])
        link.drain(C, wait=True)

    rt = Runtime(seq, [in_ty, out_ty])
    rt.add_flow(link)
    return Program(iron.get_current_device(), rt).resolve_program()


@pytest.mark.parametrize("route", ["flow", "routed", "packet"])
def test_flow_shim_to_shim(route):
    host = np.random.default_rng(7).integers(-(2**31), 2**31 - 1, N, dtype=np.int32)
    a = iron.tensor(host, dtype=np.int32, device="npu")
    c = iron.zeros(N // 2, dtype=np.int32, device="npu")
    shim_to_shim(a, c, route=route)
    c.to("cpu")
    np.testing.assert_array_equal(c.numpy(), host[::2])
