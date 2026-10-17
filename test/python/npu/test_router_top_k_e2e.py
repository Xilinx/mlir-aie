# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# RUN: %run_on_npu2_hrx% %pytest %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings
"""Route with sample_select and router_top_k, and match router_top_k_ref exactly.

As a MoE router runs them: one select core reduces a row of router logits to
a summary of its top_k candidates in one call, and one router core turns the
summary into the top_k expert ids and a softmax over their values, a real
weight at every even index and a zero at every odd one (or, packed, every
index a weight); without normalize the router reads the row as well, and each
weight is a share of the whole row's softmax. A dispatch routes
several rows, so the select core's state is also checked to reset itself
between them.

The rows are chosen for the places a ranking can go wrong: ties at the k-th
value (the lowest index wins), a whole row of one value,
signed zeros, rows that are mostly -inf, and the largest value last.
"""

import aie.iron as iron
import numpy as np
import pytest
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
from aie.iron.kernels import moe, sample
from ml_dtypes import bfloat16

I32 = np.dtype[np.int32]


@jit
def route_rows(
    logits_in: In,
    ids_out: Out,
    weights_out: Out,
    *,
    experts: CompileTime[int] = 32,
    top_k: CompileTime[int] = 8,
    rows: CompileTime[int] = 1,
    normalize: CompileTime[bool] = True,
    packed: CompileTime[bool] = False,
):
    """``rows`` routings, one row of ``experts`` logits each."""
    select = sample.sample_select(slice_size=experts, chunk=experts, k_max=top_k)
    router = moe.router_top_k(
        slice_size=experts, top_k=top_k, normalize=normalize, packed=packed
    )
    width = (1 if packed else 2) * top_k
    words = sample.summary_words(experts, top_k)

    of_logits = ObjectFifo(
        np.ndarray[(experts,), np.dtype[bfloat16]], name="x", depth=2
    )
    of_summary = ObjectFifo(np.ndarray[(words,), I32], name="summary", depth=1)
    of_ids = ObjectFifo(np.ndarray[(top_k,), I32], name="ids", depth=1)
    of_weights = ObjectFifo(
        np.ndarray[(width,), np.dtype[bfloat16]], name="weights", depth=1
    )
    # sample_select reads only the row's top_k field.
    row = Buffer(
        np.ndarray[(sample.ROW_WORDS,), I32],
        initial_value=sample.draw_row(1.0, top_k, 0),
        name="row",
    )
    state = Buffer(
        np.ndarray[(sample.SELECT_STATE_WORDS,), I32],
        initial_value=np.zeros(sample.SELECT_STATE_WORDS, dtype=np.int32),
        name="select_state",
    )

    def select_body(of_x, of_sum, kernel, row, state):
        for _ in range_(rows):
            x = of_x.acquire(1)
            summary = of_sum.acquire(1)
            kernel(x, row, state, summary)
            of_x.release(1)
            of_sum.release(1)

    def router_body(of_sum, of_i, of_w, kernel, of_x):
        for _ in range_(rows):
            summary = of_sum.acquire(1)
            ids = of_i.acquire(1)
            weights = of_w.acquire(1)
            if of_x is None:
                kernel(summary, ids, weights)
            else:
                x = of_x.acquire(1)
                kernel(summary, x, ids, weights)
                of_x.release(1)
            of_sum.release(1)
            of_i.release(1)
            of_w.release(1)

    # Without normalize the router reads the row too: the logits go to both.
    whole_row = None if normalize else of_logits.cons()
    workers = [
        Worker(select_body, [of_logits.cons(), of_summary.prod(), select, row, state]),
        Worker(
            router_body,
            [of_summary.cons(), of_ids.prod(), of_weights.prod(), router, whole_row],
        ),
    ]
    host = [
        np.ndarray[(rows * experts,), np.dtype[bfloat16]],
        np.ndarray[(rows * top_k,), I32],
        np.ndarray[(rows * width,), np.dtype[bfloat16]],
    ]

    def sequence(logits_h, ids_h, weights_h, x, ids, weights):
        x.fill(logits_h)
        ids.drain(ids_h, wait=True)
        weights.drain(weights_h, wait=True)

    rt = Runtime(sequence, [*host, of_logits.prod(), of_ids.cons(), of_weights.cons()])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def _rows(experts, top_k, seed):
    """Rows of logits, one per shape below."""
    rng = np.random.default_rng(seed)
    rows = [rng.normal(0, 2, experts) for _ in range(4)]
    # Ties at the k-th value, some below the k-th index and some above.
    logits = rng.normal(0, 0.5, experts)
    tau = np.sort(logits)[::-1][top_k // 2]
    logits[rng.choice(experts, top_k, replace=False)] = tau
    rows.append(logits)
    # One value everywhere: the first top_k indices win.
    rows.append(np.full(experts, 1.5))
    # Signed zeros are one value.
    logits = np.where(rng.random(experts) < 0.5, -0.0, 0.0)
    logits[rng.choice(experts, 3, replace=False)] = 1.0
    rows.append(logits)
    # Exactly top_k finite logits, the rest -inf.
    logits = np.full(experts, -np.inf)
    logits[rng.choice(experts, top_k, replace=False)] = rng.normal(0, 1, top_k)
    rows.append(logits)
    # Falling in index order, the largest value last among the winners.
    rows.append(-np.arange(experts) / 64.0)
    rows.append(np.arange(experts) / 16.0)
    return np.stack(rows).astype(bfloat16)


@pytest.mark.parametrize(
    "experts,top_k,normalize,packed",
    [
        # Granite-3.1-MoE's router: 8 of 32 experts.
        (32, 8, True, False),
        # 70 = 2 * 32 + 6: the select call ends in a scalar tail.
        (70, 4, True, False),
        (64, 16, True, False),
        # Granite 4's 6 of 64, packed.
        (64, 6, True, True),
        # Qwen2-MoE's 4 of 60, a share of the whole row's softmax.
        (60, 4, False, False),
        (60, 4, False, True),
        # Qwen3-MoE's 8 of 128.
        (128, 8, True, True),
    ],
)
def test_device_routing_matches_router_top_k_ref(experts, top_k, normalize, packed):
    logits = _rows(experts, top_k, seed=experts + top_k)
    rows = len(logits)
    width = (1 if packed else 2) * top_k
    ids = iron.full((rows * top_k,), -1, dtype=np.int32)
    weights = iron.full((rows * width,), -1.0, dtype=bfloat16)
    route_rows(
        iron.tensor(logits.reshape(-1), dtype=bfloat16),
        ids,
        weights,
        experts=experts,
        top_k=top_k,
        rows=rows,
        normalize=normalize,
        packed=packed,
    )
    got_ids = ids.numpy().reshape(rows, top_k)
    got_weights = weights.numpy().reshape(rows, width)
    for r in range(rows):
        want_ids, want_weights = moe.router_top_k_ref(
            logits[r], top_k, normalize=normalize, packed=packed
        )
        np.testing.assert_array_equal(got_ids[r], want_ids, err_msg=f"row {r} ids")
        np.testing.assert_array_equal(
            got_weights[r].view(np.uint16),
            want_weights.view(np.uint16),
            err_msg=f"row {r} weights",
        )
