# test_set_lock_logical_tile.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""On-device test of aiex.set_lock on a lock whose tile is not yet placed.

A Worker waits on a lock the runtime sequence sets: a WorkerRuntimeBarrier,
or a Lock with an explicit id. On an unplaced Worker the lock's tile is placed
only by aie-place-tiles; a set_lock aimed at the wrong tile would leave the
Worker waiting, and the dispatch would time out.
"""

import aie.iron as iron
import numpy as np
import pytest
from aie.dialects._aie_enum_gen import AIETileType
from aie.iron import (
    CompileTime,
    In,
    Lock,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    WorkerRuntimeBarrier,
)
from aie.iron.controlflow import range_
from aie.iron.device import Tile

N = 256
TILE = 16


@iron.jit
def gated_add(
    a: In, b: Out, *, placed: CompileTime[bool], explicit_id: CompileTime[bool]
):
    tile_ty = np.ndarray[(TILE,), np.dtype[np.int32]]
    ty = np.ndarray[(N,), np.dtype[np.int32]]
    of_in, of_out = ObjectFifo(tile_ty, name="i"), ObjectFifo(tile_ty, name="o")
    tile = Tile(0, 3) if placed else Tile(tile_type=AIETileType.CoreTile)
    gate = Lock(tile, lock_id=5, init=0) if explicit_id else WorkerRuntimeBarrier()

    def core(cin, cout, gate):
        if explicit_id:
            gate.acquire_exact(1)
            gate.release(0)
        else:
            gate.wait_for_value(1)
        for _ in range_(N // TILE):
            x, y = cin.acquire(1), cout.acquire(1)
            for i in range_(TILE):
                y[i] = x[i] + 3
            cin.release(1)
            cout.release(1)

    def seq(A, B, hin, hout):
        hin.fill(A)
        gate.set(1)
        hout.drain(B, wait=True)

    worker = Worker(core, [of_in.cons(), of_out.prod(), gate], tile=tile)
    rt = Runtime(seq, [ty, ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


@pytest.mark.parametrize("placed", [False, True])
@pytest.mark.parametrize("explicit_id", [False, True])
def test_the_sequence_releases_a_worker_on_any_tile(placed, explicit_id):
    host = np.random.default_rng(4).integers(-(2**20), 2**20, N, dtype=np.int32)
    a = iron.tensor(host, dtype=np.int32, device="npu")
    b = iron.zeros(N, dtype=np.int32, device="npu")
    gated_add(a, b, placed=placed, explicit_id=explicit_id)
    b.to("cpu")
    np.testing.assert_array_equal(b.numpy(), host + 3)
