# test_objectfifo_join_of_joins.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""On-device test of a join of joins: eight lanes into one stream.

A memtile takes six input channels, so eight lanes gather in two levels: two
memtiles each join four, and a third joins their outputs, each a fifo that is
one link's output and the next link's input. Every round, lane ``i`` carries
the ``i``-th eighth of the round, in two objects; the drained stream must be
the input, element for element.
"""

import aie.iron as iron
import numpy as np
import pytest
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime

LANES, SLOT, OBJ = 8, 256, 128
ROUND = LANES * SLOT


def vector(n):
    return np.ndarray[(n,), np.dtype[np.int32]]


@iron.jit
def join_of_joins(a: In, c: Out, *, rounds: CompileTime[int] = 4):
    out = ObjectFifo(vector(ROUND), depth=2)
    halves = out.prod().join([0, ROUND // 2], obj_types=[vector(ROUND // 2)] * 2)
    lanes = [
        lane
        for half in halves
        for lane in half.prod().join(
            [k * SLOT for k in range(LANES // 2)], obj_types=[vector(OBJ)] * 4
        )
    ]

    def sequence(A, C, ins, drained):
        for i, lane in enumerate(ins):
            lane.fill(
                A,
                tap=TensorAccessPattern(
                    (rounds * ROUND,), i * SLOT, [rounds, SLOT], [ROUND, 1]
                ),
            )
        drained.drain(C, wait=True)

    rt = Runtime(
        sequence,
        [
            vector(rounds * ROUND),
            vector(rounds * ROUND),
            [lane.prod() for lane in lanes],
            out.cons(),
        ],
    )
    return Program(iron.get_current_device(), rt).resolve_program()


@pytest.mark.parametrize("rounds", [1, 4])
def test_join_of_joins_keeps_the_element_order(rounds):
    n = rounds * ROUND
    host = np.random.default_rng(3).integers(-(2**31), 2**31 - 1, n, dtype=np.int32)
    a = iron.tensor(host, dtype=np.int32, device="npu")
    c = iron.zeros(n, dtype=np.int32, device="npu")
    join_of_joins(a, c, rounds=rounds)
    c.to("cpu")
    np.testing.assert_array_equal(c.numpy(), host)
