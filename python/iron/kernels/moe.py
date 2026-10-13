# kernels/moe.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""A mixture-of-experts router's top-k selection on the device.

``router_top_k`` reads one column's ``sample_select`` summary (``sample.py``,
``k_max == top_k``, one call reduces the whole row) and writes the ``top_k``
expert ids and their softmax weights, deterministically: where
``sample_combine`` draws one token stochastically from every candidate at or
above its threshold, routing instead takes exactly the ``top_k`` candidates
the summary already holds (the entries strictly above tau, then ties at tau
in index order, matching ``torch.topk``'s tie-break) and a softmax over their
values alone.
"""

import numpy as np
from aie.iron.kernel import ExternalFunction
from aie.utils.compile.jit.markers import In, Out
from ml_dtypes import bfloat16

from ._common import KernelContract, Trace, _kernel_source, _make_extern
from .sample import K_MAX_LIMIT, exp64_ref, order_keys, summary_words

_STRICT_FP = ["-ffp-contract=off", "-fno-fast-math"]


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
    if slice_size < top_k:
        raise ValueError(
            f"router_top_k: slice_size ({slice_size}) must be at least top_k ({top_k})"
        )
    if top_k > K_MAX_LIMIT:
        raise ValueError(
            f"router_top_k: top_k {top_k} is above {K_MAX_LIMIT}, the largest "
            "sample_select summary verified on hardware"
        )
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

    Exact top-``top_k`` of ``logits`` (ties broken by ascending index, as
    ``torch.topk`` does), then a softmax over the selected values: a
    soft-float IEEE float32 subtraction against the row's max (what
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
