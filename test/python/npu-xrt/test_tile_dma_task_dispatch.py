# test_tile_dma_task_dispatch.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""On-device test of a mem tile descriptor rebuilt per dispatch.

The host stages MAX elements into a mem tile Buffer, then a tile_dma_task on
the mem tile's MM2S reads back a window whose start and length are
DispatchTime values. One compiled design serves every window. A lock pair
hands the buffer from the S2MM to the MM2S task, as a resident operand would be
handed to a compute tile.
"""

import aie.iron as iron
import numpy as np
import pytest
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir
from aie.extras.dialects import arith
from aie.helpers.util import np_dtype_to_mlir_type
from aie.iron import (
    Acquire,
    Buffer,
    DispatchTime,
    Flow,
    In,
    Lock,
    Out,
    Program,
    Release,
    Runtime,
    tile_dma_task,
)
from aie.iron.device import Tile

CHUNK = 128
MAX_CHUNKS = 8
MAX = CHUNK * MAX_CHUNKS


def _window(start, chunks, dtype, explicit_len):
    host_ty = np.ndarray[(MAX,), np.dtype[np.int32]]
    shim = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
    mem = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    resident = Buffer(type=host_ty, tile=mem, name="resident")
    empty = Lock(tile=mem, init=1, name="empty")
    full = Lock(tile=mem, init=0, name="full")
    into = Flow(shim, mem, src_channel=0, dst_channel=0)
    out = Flow(mem, shim, src_channel=0, dst_channel=0)

    def seq(A, C, s, n):
        chunk = arith.constant(CHUNK, np_dtype_to_mlir_type(dtype))
        length = n * chunk if explicit_len else None
        into.fill(A)
        tile_dma_task(
            mem,
            DMAChannelDir.S2MM,
            into.endpoint(mem),
            resident,
            acquire=Acquire(empty),
            release=Release(full),
        )
        tile_dma_task(
            mem,
            DMAChannelDir.MM2S,
            out.endpoint(mem),
            resident,
            sizes=[1, 1, n, CHUNK],
            strides=[0, 0, CHUNK, 1],
            offset=s * chunk,
            transfer_len=length,
            acquire=Acquire(full),
            release=Release(empty),
        )
        out.drain(
            C,
            sizes=[1, 1, n, CHUNK],
            strides=[0, 0, CHUNK, 1],
            transfer_len=length,
            wait=True,
        )

    rt = Runtime(seq, [host_ty, host_ty, start, chunks])
    rt.add_flow(into)
    rt.add_flow(out)
    rt.add_lock(empty)
    rt.add_lock(full)
    rt.add_buffer(resident)
    return Program(iron.get_current_device(), rt).resolve_program()


@iron.jit
def window(
    a: In,
    c: Out,
    *,
    start: DispatchTime[np.int64] = 0,
    chunks: DispatchTime[np.int64] = 1,
):
    return _window(start, chunks, np.int64, explicit_len=True)


# i32 scalars are widened to the i64 sizes, and the omitted lengths default to
# the product of the sizes.
@iron.jit
def window_i32(
    a: In,
    c: Out,
    *,
    start: DispatchTime[np.int32] = 0,
    chunks: DispatchTime[np.int32] = 1,
):
    return _window(start, chunks, np.int32, explicit_len=False)


@pytest.mark.parametrize("jitted", [window, window_i32], ids=["i64", "i32"])
def test_tile_dma_task_dispatch_window(jitted):
    design = jitted.specialize()
    a = iron.tensor(
        np.random.default_rng(0).integers(0, 2**16, size=(MAX,), dtype=np.int32),
        dtype=np.int32,
        device="npu",
    )
    first_kernel = None
    for start, chunks in ((0, 1), (2, 3), (0, MAX_CHUNKS), (5, 3), (7, 1)):
        c = iron.zeros((MAX,), dtype=np.int32, device="npu")
        design(a, c, start=start, chunks=chunks)
        expected = np.zeros((MAX,), dtype=np.int32)
        n = chunks * CHUNK
        expected[:n] = a.numpy()[start * CHUNK : start * CHUNK + n]
        np.testing.assert_array_equal(c.numpy(), expected)

        assert len(design._kernel_cache) == 1
        kernel = next(iter(design._kernel_cache.values()))
        if first_kernel is None:
            first_kernel = kernel
        assert kernel is first_kernel
