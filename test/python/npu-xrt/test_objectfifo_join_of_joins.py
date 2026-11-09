# test_objectfifo_join_of_joins.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""On-device test of a join of joins: lanes into one stream.

The lanes gather in two levels: two joins each take half, and a third joins
their outputs, each a fifo that is one link's output and the next link's
input. Eight lanes need a second memtile, past one's six input channels;
four may put all three links on one memtile, the chained fifos looping
through its stream switch. Every round, lane ``i`` carries the ``i``-th slot
of the round, in two objects; the drained stream must be the input, element
for element.
"""

import aie.iron as iron
import numpy as np
import pytest
from aie.dialects._aie_enum_gen import AIETileType
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime
from aie.iron.device import Tile

SLOT, OBJ = 256, 128


def vector(n):
    return np.ndarray[(n,), np.dtype[np.int32]]


@iron.jit
def join_of_joins(
    a: In,
    c: Out,
    *,
    rounds: CompileTime[int] = 4,
    lanes: CompileTime[int] = 8,
    one_tile: CompileTime[bool] = False,
):
    block = lanes * SLOT
    on = {"tile": Tile(1, 1, tile_type=AIETileType.MemTile)} if one_tile else {}
    out = ObjectFifo(vector(block), depth=2)
    halves = out.prod().join([0, block // 2], obj_types=[vector(block // 2)] * 2, **on)
    ends = [
        lane
        for half in halves
        for lane in half.prod().join(
            [k * SLOT for k in range(lanes // 2)],
            obj_types=[vector(OBJ)] * (lanes // 2),
            **on,
        )
    ]

    def sequence(A, C, ins, drained):
        for i, lane in enumerate(ins):
            lane.fill(
                A,
                tap=TensorAccessPattern(
                    (rounds * block,), i * SLOT, [rounds, SLOT], [block, 1]
                ),
            )
        drained.drain(C, wait=True)

    rt = Runtime(
        sequence,
        [
            vector(rounds * block),
            vector(rounds * block),
            [lane.prod() for lane in ends],
            out.cons(),
        ],
    )
    return Program(iron.get_current_device(), rt).resolve_program()


@pytest.mark.parametrize("rounds", [1, 4])
@pytest.mark.parametrize("lanes, one_tile", [(8, False), (4, True)])
def test_join_of_joins_keeps_the_element_order(rounds, lanes, one_tile):
    n = rounds * lanes * SLOT
    host = np.random.default_rng(3).integers(-(2**31), 2**31 - 1, n, dtype=np.int32)
    a = iron.tensor(host, dtype=np.int32, device="npu")
    c = iron.zeros(n, dtype=np.int32, device="npu")
    join_of_joins(a, c, rounds=rounds, lanes=lanes, one_tile=one_tile)
    c.to("cpu")
    np.testing.assert_array_equal(c.numpy(), host)
