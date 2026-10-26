# test_runtime_merge.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""On-device test of Runtime.merge: steps that run back to back as one sequence.

Each step has its own Worker and ObjectFifos on one configuration. Two steps
add 7 and double; two others forward one DispatchTime parameter, so the
merged sequence takes it once and both steps read it.
"""

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    DeviceConfiguration,
    DispatchTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import Tile

N = 1024
TILE = 16


def step(op, col, name, tap=None, forward=()):
    """A Worker on column ``col`` applying ``op``, and the Runtime that feeds it.

    With ``tap`` the Runtime moves one ``TILE`` of the tensors, at the index
    ``tap`` computes from the forwarded values.
    """
    tile_ty = np.ndarray[(TILE,), np.dtype[np.int32]]
    tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]
    of_in = ObjectFifo(tile_ty, name=f"{name}_in")
    of_out = ObjectFifo(tile_ty, name=f"{name}_out")
    count = N // TILE if tap is None else 1

    def core(cin, cout):
        for _ in range_(count):
            x = cin.acquire(1)
            y = cout.acquire(1)
            for i in range_(TILE):
                y[i] = op(x[i])
            cin.release(1)
            cout.release(1)

    def seq(a, b, hin, hout, *_):
        part = None if tap is None else tap()
        hin.fill(a, tap=part)
        hout.drain(b, tap=part, wait=True)

    worker = Worker(core, [of_in.cons(), of_out.prod()], tile=Tile(col, 2))
    fn_args = [tensor_ty, tensor_ty, of_in.prod(), of_out.cons(), *forward]
    return worker, Runtime(seq, fn_args)


@iron.jit
def add_then_double(a: In, b: Out, c: In, d: Out):
    tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]
    w1, r1 = step(lambda v: v + 7, 0, "add")
    w2, r2 = step(lambda v: v * 2, 1, "double")
    merged = Runtime.merge([r1, r2], name="steps")
    assert merged.arg_types == [tensor_ty] * 4
    config = DeviceConfiguration(
        "pack", iron.get_current_device(), workers=[w1, w2], runtimes=[merged]
    )
    return Program.compose([config], entry=merged).resolve_program()


@iron.jit
def two_tiles(a: In, b: Out, c: In, d: Out, *, start: DispatchTime[np.int32] = 0):
    tiles = TensorAccessPattern.full((N,)).split(0, TILE)
    w1, r1 = step(
        lambda v: v + 7, 0, "add", tap=lambda: tiles[start * 2 + 1], forward=[start]
    )
    w2, r2 = step(
        lambda v: v * 2, 1, "double", tap=lambda: tiles[start], forward=[start]
    )
    merged = Runtime.merge([r1, r2], name="steps")
    config = DeviceConfiguration(
        "pack", iron.get_current_device(), workers=[w1, w2], runtimes=[merged]
    )
    return Program.compose([config], entry=merged).resolve_program()


def test_merged_steps_run_in_one_sequence():
    rng = np.random.default_rng(0)
    first = rng.integers(-(2**20), 2**20, N, dtype=np.int32)
    second = rng.integers(-(2**20), 2**20, N, dtype=np.int32)
    a = iron.tensor(first, dtype=np.int32, device="npu")
    c = iron.tensor(second, dtype=np.int32, device="npu")
    b = iron.zeros(N, dtype=np.int32, device="npu")
    d = iron.zeros(N, dtype=np.int32, device="npu")
    add_then_double(a, b, c, d)
    b.to("cpu")
    d.to("cpu")
    np.testing.assert_array_equal(b.numpy(), first + 7)
    np.testing.assert_array_equal(d.numpy(), second * 2)


def test_merged_steps_read_one_forwarded_dispatch_value():
    rng = np.random.default_rng(1)
    first = rng.integers(-(2**20), 2**20, N, dtype=np.int32)
    second = rng.integers(-(2**20), 2**20, N, dtype=np.int32)
    a = iron.tensor(first, dtype=np.int32, device="npu")
    c = iron.tensor(second, dtype=np.int32, device="npu")
    for start in (0, 3, 1):
        b = iron.zeros(N, dtype=np.int32, device="npu")
        d = iron.zeros(N, dtype=np.int32, device="npu")
        two_tiles(a, b, c, d, start=np.int32(start))
        b.to("cpu")
        d.to("cpu")
        added = np.zeros(N, dtype=np.int32)
        lo = (2 * start + 1) * TILE
        added[lo : lo + TILE] = first[lo : lo + TILE] + 7
        doubled = np.zeros(N, dtype=np.int32)
        doubled[start * TILE : (start + 1) * TILE] = (
            second[start * TILE : (start + 1) * TILE] * 2
        )
        np.testing.assert_array_equal(b.numpy(), added)
        np.testing.assert_array_equal(d.numpy(), doubled)
