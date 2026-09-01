# test_dispatch_time_scalar.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# RUN: %run_on_npu2_hrx% %pytest %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings

# End-to-end DispatchTime[T]: one compiled design, called with different
# runtime scalar values. `n_tiles` sizes both the fill and the drain, so a
# single BD carries the tile count -- no host-side range_ loop is needed.

import aie.iron as iron
import numpy as np
import pytest
from aie.extras.dialects import arith
from aie.helpers.util import np_dtype_to_mlir_type
from aie.iron import (
    CompileTime,
    DispatchTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    TaskGroup,
    Worker,
)
from aie.iron.controlflow import range_

TILE_SIZE = 256
MAX_TILES = 8


@iron.jit
def dyn_copy(
    a: In,
    b: Out,
    n_tiles: DispatchTime[np.int32],
    *,
    tile_size: CompileTime[int] = TILE_SIZE,
    max_tiles: CompileTime[int] = MAX_TILES,
):
    tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
    max_ty = np.ndarray[(max_tiles * tile_size,), np.dtype[np.int32]]

    of_in = ObjectFifo(tile_ty, name="of_in", depth=2)
    of_out = ObjectFifo(tile_ty, name="of_out", depth=2)

    def core_fn(in_cons, out_prod):
        elem_in = in_cons.acquire(1)
        elem_out = out_prod.acquire(1)
        for i in range_(tile_size):
            elem_out[i] = elem_in[i]
        in_cons.release(1)
        out_prod.release(1)

    worker = Worker(core_fn, [of_in.cons(), of_out.prod()])

    def seq(a_h, b_h, n, in_prod, out_cons):
        i32 = np_dtype_to_mlir_type(np.int32)
        i64 = np_dtype_to_mlir_type(np.int64)
        # dma_bd sizes/strides are i64 (DynamicIndexList); offset/len are i32
        # -- matches matmul_whole_array_dynamic.py's dyn/static width split.
        n64 = arith.extsi(i64, n)
        transfer_len = n * arith.constant(tile_size, i32)

        tg = TaskGroup()
        out_cons.drain(
            b_h,
            sizes=[1, 1, n64, tile_size],
            strides=[0, 0, tile_size, 1],
            offset=0,
            transfer_len=transfer_len,
            wait=True,
            group=tg,
        )
        in_prod.fill(
            a_h,
            sizes=[1, 1, n64, tile_size],
            strides=[0, 0, tile_size, 1],
            offset=0,
            transfer_len=transfer_len,
            group=tg,
        )
        tg.finish()

    rt = Runtime(seq, [max_ty, max_ty, np.int32, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def _random_tiles(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**16, size=(MAX_TILES * TILE_SIZE,), dtype=np.int32)


def test_dispatch_time_scalar_varies_without_recompile():
    """Two calls with different n_tiles must both be correct.

    And must share one compiled artifact (no recompile between them).
    """
    a1 = iron.tensor(_random_tiles(seed=1), dtype=np.int32, device="npu")
    b1 = iron.zeros((MAX_TILES * TILE_SIZE,), dtype=np.int32, device="npu")
    dyn_copy(a1, b1, 3)
    assert np.array_equal(b1.numpy()[: 3 * TILE_SIZE], a1.numpy()[: 3 * TILE_SIZE])

    # One cache entry per compiled artifact. _kernel_dir cannot vary with a
    # dispatch value by construction, so asserting on it would pass even if
    # every value rebuilt; the kernel cache is what actually shows reuse.
    kernels_after_first_call = len(dyn_copy._kernel_cache)

    a2 = iron.tensor(_random_tiles(seed=2), dtype=np.int32, device="npu")
    b2 = iron.zeros((MAX_TILES * TILE_SIZE,), dtype=np.int32, device="npu")
    dyn_copy(a2, b2, 6)
    assert np.array_equal(b2.numpy()[: 6 * TILE_SIZE], a2.numpy()[: 6 * TILE_SIZE])

    # A second, different value must reuse the first call's kernel rather than
    # add one -- the whole point of DispatchTime[T] is "one compile, many
    # values", and passing values positionally must not defeat it.
    assert len(dyn_copy._kernel_cache) == kernels_after_first_call


def test_dispatch_time_scalar_repeated_same_value():
    """Repeat calls with the *same* n_tiles must keep working.

    The call above varies n_tiles, which changes the kernel cache key and so
    never exercises the in-memory kernel-cache hit. That path used to validate
    a cached kernel via ``Path(kernel.insts_path)``, which a dispatch design
    leaves as ``None`` -- so the second identical call raised TypeError.
    """
    a = iron.tensor(_random_tiles(seed=4), dtype=np.int32, device="npu")
    for _ in range(3):
        b = iron.zeros((MAX_TILES * TILE_SIZE,), dtype=np.int32, device="npu")
        dyn_copy(a, b, 3)
        assert np.array_equal(b.numpy()[: 3 * TILE_SIZE], a.numpy()[: 3 * TILE_SIZE])
        # Beyond n_tiles the buffer must be untouched -- proves the runtime
        # scalar, not a fixed compiled-in bound, sized the transfer.
        assert np.all(b.numpy()[3 * TILE_SIZE :] == 0)


def test_dispatch_time_scalar_rejects_missing_value():
    from aie.utils.hostruntime.hostruntime import HostRuntimeError

    a = iron.tensor(_random_tiles(seed=3), dtype=np.int32, device="npu")
    b = iron.zeros((MAX_TILES * TILE_SIZE,), dtype=np.int32, device="npu")
    with pytest.raises(HostRuntimeError, match="dispatch scalar mismatch"):
        dyn_copy(a, b)
