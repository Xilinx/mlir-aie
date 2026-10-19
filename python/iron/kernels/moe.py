# kernels/moe.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Mixture of experts kernels: the router's top-k selection and the experts' FFN.

``router_top_k`` reads one column's ``sample_select`` summary (``sample.py``,
``k_max == top_k``, one call reduces the whole row) and writes the ``top_k``
expert ids and their softmax weights, deterministically: where
``sample_combine`` draws one token stochastically from every candidate at or
above its threshold, routing instead takes exactly the ``top_k`` candidates
the summary already holds (the entries strictly above tau in index order,
then the lowest-index ties at tau, not ranked by value) and a softmax over
their values alone. That softmax is the weighting of routers that normalize
over the selected experts (Granite-MoE, Mixtral, ``norm_topk_prob=True``).
"""

from functools import partial

import numpy as np
from aie.iron.kernel import ExternalFunction
from aie.utils.compile.jit.markers import In, Out
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import KernelContract, Param, Trace, _kernel_source, _make_extern
from .core import conv_even
from .sample import (
    _STRICT_FP,
    _STRICT_FP_ROLLED,
    _check_slice,
    _positive,
    exp64_ref,
    order_keys,
    summary_words,
)


def router_top_k(*, slice_size=32, top_k=8) -> ExternalFunction:
    """One column's router: ``sample_select``'s summary to ``top_k`` ids and weights.

    ``router_top_k(summary, ids, weights)`` takes the ``summary_words(slice_size,
    top_k)`` int32 words a ``sample_select(slice_size=slice_size, chunk=slice_size,
    k_max=top_k)`` call produces for the whole row in one pass, and writes
    ``top_k`` int32 ``ids`` and ``2 * top_k`` bf16 ``weights``: ``ids[i]``
    pairs with ``weights[2 * i]``, ``weights[2 * i + 1]`` a zeroed, never
    meaningful pad word. The padding keeps every real weight at a
    4-byte-aligned bf16 index, which a graph's ``Copy`` of one selected
    weight into another buffer needs (found tracing a real
    ``iron/common/image/fusion.py`` alignment failure, not assumed: the
    allocator always places a buffer's own base 64-byte aligned, so an
    unpadded ``weights[i]`` at odd ``i`` is the one place the misalignment
    could come from).
    """
    _positive("router_top_k", slice_size=slice_size, top_k=top_k)
    _check_slice("router_top_k", slice_size, top_k)
    return _make_extern(
        "router_top_k",
        _kernel_source("moe/router_top_k.cc"),
        [
            np.ndarray[(summary_words(slice_size, top_k),), np.dtype[np.int32]],
            np.ndarray[(top_k,), np.dtype[np.int32]],
            np.ndarray[(2 * top_k,), np.dtype[bfloat16]],
        ],
        compile_flags=_STRICT_FP
        + [f"-DSAMPLE_SLICE={slice_size}", f"-DSAMPLE_K_MAX={top_k}"],
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out, Out),
        ),
    )


def router_top_k_ref(logits, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """``(ids, weights)`` as the device's ``router_top_k`` writes one row.

    Bit for bit: see ``router_top_k.cc``. ``weights`` is ``2 * top_k`` wide,
    a real weight at every even index (``weights[2 * i]`` pairs with
    ``ids[i]``) and a zeroed pad word at every odd one.

    Exact top-``top_k`` of ``logits``, the lowest indices winning a tie at
    the ``top_k``-th value. ``ids`` holds the entries above that value in
    index order, then the tied ones in index order, not ranked by value as
    ``torch.topk(sorted=True)`` is. Then a softmax over the selected values:
    a soft-float IEEE float32 subtraction against the row's max (what
    ``__subsf3`` gives the device, where Peano's native ``-`` would not),
    ``exp64`` (``sample.py``'s bit-exact float64 exponential, the same
    primitive ``sample_combine`` uses) and a plain float64 sum and divide,
    rounded once to bf16.
    """
    row = np.ascontiguousarray(logits, dtype=bfloat16).reshape(-1)
    keys = order_keys(row).astype(np.int64)
    n = keys.size
    if not 1 <= top_k <= n:
        raise ValueError(f"router_top_k_ref: top_k ({top_k}) must be in [1, {n}]")
    tau = np.partition(keys, n - top_k)[n - top_k]
    above = np.flatnonzero(keys > tau)  # ascending index, fewer than top_k
    ties = np.flatnonzero(keys == tau)  # ascending index
    needed = top_k - above.size
    ids = np.concatenate([above, ties[:needed]]).astype(np.int32)
    max_value = row[int(np.argmax(keys))].astype(np.float32)
    values = row[ids].astype(np.float32)
    shifted = (values - max_value).astype(np.float32)
    w = exp64_ref(shifted.astype(np.float64))
    packed = (w / w.sum()).astype(np.float32).astype(bfloat16)
    weights = np.zeros(2 * top_k, dtype=bfloat16)
    weights[0::2] = packed
    return ids, weights


def _experts_steps(hidden, width, rows, cols):
    """Tiles per expert: gate_up's, then the expert's total."""
    gate_up = (2 * width // rows) * (hidden // cols)
    return gate_up, gate_up + (hidden // rows) * (width // cols)


def _experts_row(x, weights, ids):
    """Pack one row buffer: x, the padded weights, then the int32 ids as bf16 pairs."""
    ids = np.ascontiguousarray(ids, np.int32).view(bfloat16)
    return np.concatenate([np.asarray(x, bfloat16), np.asarray(weights, bfloat16), ids])


def experts_sample(rng, calls, *, hidden, width, num_experts, top_k, rows, cols):
    """Return a row per ``num_experts`` experts' tiles, as a design passes it; random tiles."""
    row_steps = num_experts * _experts_steps(hidden, width, rows, cols)[1]
    packed = []
    for _ in range(-(-calls // row_steps)):
        weights = np.zeros(2 * top_k, np.float32)
        weights[::2] = rng.random(top_k)
        ids = rng.choice(num_experts, top_k, replace=False)
        packed.append(_experts_row(rng.standard_normal(hidden) * 0.5, weights, ids))
    rows_in = np.repeat(np.stack(packed), row_steps, axis=0)[:calls]
    tiles = rng.standard_normal((calls, rows * cols)) * 0.1
    return [rows_in, tiles.astype(bfloat16)]


def experts_ref(row, tile, *, hidden, width, num_experts, top_k, rows, cols):
    """Numpy reference for [`experts`][iron.kernels.moe.experts]: ``y`` after each call, call ``c`` being step ``c``."""
    gate_up_steps, expert_steps = _experts_steps(hidden, width, rows, cols)
    row = np.ascontiguousarray(row, bfloat16).reshape(-1, hidden + 4 * top_k)
    ids = row[:, hidden + 2 * top_k :].copy().view(np.int32)
    row = row.astype(np.float32)
    tile = np.asarray(tile, np.float32).reshape(-1, rows, cols)
    y = np.zeros((len(row), hidden), np.float32)
    for call, (r, t) in enumerate(zip(row, tile)):
        e, s = divmod(call % (num_experts * expert_steps), expert_steps)
        if e == 0 and s == 0:
            total = np.zeros(hidden, np.float32)
        chosen = ids[call] == e
        if s == 0:
            acc = np.zeros(2 * width, np.float32)
        if chosen.any() and s < gate_up_steps:
            r0, c0 = divmod(s, hidden // cols)
            r0, c0 = r0 * rows, c0 * cols
            acc[r0 : r0 + rows] += t @ r[c0 : c0 + cols]
            if s == gate_up_steps - 1:
                g = acc[:width]
                with np.errstate(over="ignore"):
                    acc[:width] = g / (1 + np.exp(-g)) * acc[width:]
        elif chosen.any():
            weight = r[hidden : hidden + 2 * top_k : 2][chosen].sum()
            r0, c0 = divmod(s - gate_up_steps, width // cols)
            r0, c0 = r0 * rows, c0 * cols
            total[r0 : r0 + rows] += weight * (t @ acc[c0 : c0 + cols])
        y[call] = total
    return y.astype(bfloat16)


def experts(
    *,
    hidden: int = 64,
    width: int = 32,
    num_experts: int = 4,
    top_k: int = 2,
    rows: int = 8,
    cols: int = 32,
) -> ExternalFunction:
    """Mixture of experts' feed-forward, a row at a time, from two input streams.

    ``experts(row, tile, y, step)``: ``row`` is bf16 x (``hidden``), then
    the row's ``2 * top_k`` router weights as ``router_top_k`` writes them
    (chosen expert ``j``'s at ``2 * j``), then its ``top_k`` int32 ids as
    pairs of bf16 words. Step ``s`` brings tile ``s`` modulo a row's steps,
    ``rows`` x ``cols`` bf16, of one fixed stream: for each of the
    ``num_experts`` experts in order, the blocks of its ``gate_up`` (``2 *
    width`` x ``hidden``) row block by row block, then those of its ``down``
    (``hidden`` x ``width``). The kernel skips the tiles of experts the row
    did not choose. ``y`` is the row's bf16 output so far; after the row's
    last step it is the sum over ``j`` of weight ``j`` times ``down[ids[j]]
    @ (silu(g) * u)``, ``g`` and ``u`` the halves of ``gate_up[ids[j]] @
    x``, in float32 and rounded once. A shim BD repeats a dimension at most
    64 times, so at Qwen3 shapes ``rows`` is 32 or more.
    """
    geometry = dict(
        hidden=hidden,
        width=width,
        num_experts=num_experts,
        top_k=top_k,
        rows=rows,
        cols=cols,
    )
    _positive("experts", **geometry)
    if (2 * width) % rows or hidden % rows or hidden % cols or width % cols:
        raise ValueError(
            "experts: rows must divide 2 * width and hidden, and cols must "
            "divide hidden and width"
        )
    if hidden % 2 or top_k > num_experts:
        raise ValueError(
            "experts: hidden must be even, so the ids are word aligned, and "
            "top_k at most num_experts"
        )
    return _make_extern(
        "experts",
        _kernel_source("moe/experts.cc"),
        [
            np.ndarray[(hidden + 4 * top_k,), np.dtype[bfloat16]],
            np.ndarray[(rows * cols,), np.dtype[bfloat16]],
            np.ndarray[(hidden,), np.dtype[bfloat16]],
            np.int32,
        ],
        compile_flags=_STRICT_FP_ROLLED
        + [
            f"-DEXPERTS_HIDDEN={hidden}",
            f"-DEXPERTS_WIDTH={width}",
            f"-DEXPERTS_EXPERTS={num_experts}",
            f"-DEXPERTS_TOP_K={top_k}",
            f"-DEXPERTS_TILE_ROWS={rows}",
            f"-DEXPERTS_TILE_COLS={cols}",
        ],
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, In, Out, Param),
            parameter_bindings=((3, 0),),
            call_index=3,
            reference=partial(experts_ref, **geometry),
            sample=partial(experts_sample, **geometry),
            setup=conv_even,
            tolerance=Tolerance(
                ulps=1,
                range_frac=1e-5,
                note="host clang build of the scalar source; the floor is for "
                "sums that cancel",
            ),
        ),
    )
