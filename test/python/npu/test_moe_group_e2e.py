# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# RUN: %run_on_npu2_hrx% %pytest %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings
"""Group a chunk's routed rows by expert with moe_group, and match
moe_group_ref exactly.

One core takes the chunk's ids twice, call_rows rows a call: the first pass
counts each expert's assignments, then the blocks are laid out, and the
second pass gives each assignment its slot. The routings are chosen for the
places a layout can go wrong: every row routed alike (the most blocks), an
expert no row picks, a bound mid-call and none at all, and ids outside the
experts.
"""

import aie.dialects.index as index
import aie.iron as iron
import numpy as np
import pytest
from aie.dialects.aie import T
from aie.iron import (
    Buffer,
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    jit,
)
from aie.iron.controlflow import range_
from aie.iron.kernels import moe

I32 = np.dtype[np.int32]


@jit
def group_rows(
    ids_in: In,
    blocks_out: Out,
    where_out: Out,
    *,
    rows: CompileTime[int] = 256,
    call_rows: CompileTime[int] = 32,
    top_k: CompileTime[int] = 8,
    experts: CompileTime[int] = 32,
    tile: CompileTime[int] = 32,
    valid: CompileTime[int] = 256,
):
    """``rows`` routed rows of ``top_k`` ids grouped, the first ``valid``."""
    group = moe.moe_group(
        rows=rows, call_rows=call_rows, top_k=top_k, experts=experts, tile=tile
    )
    blocks = moe.moe_group_blocks(rows=rows, top_k=top_k, experts=experts, tile=tile)
    calls = rows // call_rows
    of_ids = ObjectFifo(np.ndarray[(call_rows * top_k,), I32], name="ids", depth=2)
    of_blocks = ObjectFifo(np.ndarray[(blocks,), I32], name="blocks", depth=1)
    of_where = ObjectFifo(np.ndarray[(call_rows * top_k,), I32], name="where", depth=2)
    state = Buffer(
        np.ndarray[(2 * experts,), I32],
        initial_value=np.full(2 * experts, -1, dtype=np.int32),
        name="group_state",
    )

    def core(of_i, of_b, of_w, count, lay_out, place, state):
        for t in range_(calls):
            first = index.casts(T.i32(), t) * call_rows
            ids = of_i.acquire(1)
            count(ids, state, first, valid)
            of_i.release(1)
        b = of_b.acquire(1)
        lay_out(state, b)
        of_b.release(1)
        for t in range_(calls):
            first = index.casts(T.i32(), t) * call_rows
            ids = of_i.acquire(1)
            w = of_w.acquire(1)
            place(ids, state, w, first, valid)
            of_i.release(1)
            of_w.release(1)

    worker = Worker(
        core,
        [
            of_ids.cons(),
            of_blocks.prod(),
            of_where.prod(),
            group,
            group.blocks,
            group.where,
            state,
        ],
    )
    host = [
        np.ndarray[(rows * top_k,), I32],
        np.ndarray[(blocks,), I32],
        np.ndarray[(rows * top_k,), I32],
    ]

    def sequence(ids_h, blocks_h, where_h, i, b, w):
        i.fill(ids_h)
        i.fill(ids_h)
        b.drain(blocks_h, wait=True)
        w.drain(where_h, wait=True)

    rt = Runtime(sequence, [*host, of_ids.prod(), of_blocks.cons(), of_where.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def _routing(kind, rows, top_k, experts, rng):
    if kind == "alike":
        return np.tile(np.arange(top_k), (rows, 1))
    if kind == "outside":
        return rng.integers(-2 * experts, 2 * experts, (rows, top_k))
    # "unpicked": expert 5 is never chosen.
    choices = (
        np.delete(np.arange(experts), 5) if kind == "unpicked" else np.arange(experts)
    )
    return np.stack([rng.permutation(choices)[:top_k] for _ in range(rows)])


@pytest.mark.parametrize(
    "rows, call_rows, top_k, experts, tile, valid, kind",
    [
        # Granite-3.1-MoE's 8 of 32, a whole chunk and one cut mid-call.
        (256, 32, 8, 32, 32, 256, "random"),
        (256, 32, 8, 32, 32, 77, "random"),
        (256, 32, 8, 32, 64, 256, "alike"),
        (128, 32, 6, 64, 16, 128, "unpicked"),
        (128, 64, 4, 60, 32, 0, "random"),
        (64, 32, 2, 8, 16, 64, "outside"),
    ],
)
def test_device_grouping_matches_moe_group_ref(
    rows, call_rows, top_k, experts, tile, valid, kind
):
    rng = np.random.default_rng(rows + experts + valid)
    ids = _routing(kind, rows, top_k, experts, rng).astype(np.int32)
    blocks = moe.moe_group_blocks(rows=rows, top_k=top_k, experts=experts, tile=tile)
    got_blocks = iron.full((blocks,), -7, dtype=np.int32)
    got_where = iron.full((rows * top_k,), -7, dtype=np.int32)
    group_rows(
        iron.tensor(ids.reshape(-1), dtype=np.int32),
        got_blocks,
        got_where,
        rows=rows,
        call_rows=call_rows,
        top_k=top_k,
        experts=experts,
        tile=tile,
        valid=valid,
    )
    want_where, want_blocks = moe.moe_group_ref(
        ids, valid, experts=experts, tile=tile, blocks=blocks
    )
    np.testing.assert_array_equal(got_blocks.numpy(), want_blocks)
    np.testing.assert_array_equal(got_where.numpy().reshape(rows, top_k), want_where)
