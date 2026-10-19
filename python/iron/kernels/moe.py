# kernels/moe.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""A mixture of experts on the device: the router's top-k selection and the
experts' gated feed-forward.

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
from aie.utils.compile.jit.markers import In, InOut, Out
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import KernelContract, Trace, _kernel_source, _make_extern
from .sample import _check_slice, _positive, exp64_ref, order_keys, summary_words
from .zero import zero

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


# Scalar float32 dot products, summed in another order than numpy's; the
# floor covers terms that cancel. Measured against a host build, not a device.
_EXPERT_TOLERANCE = Tolerance.relative(
    1e-4, range_frac=1e-5, note="host clang build of the scalar source"
)


def _zero_y(fn):
    return zero(fn.arg_shape(3), np.float32)


def expert_gate_up_ref(x, w, *, rows: int, hidden: int):
    """Numpy reference for [`expert_gate_up`][iron.kernels.moe.expert_gate_up]."""
    x = np.asarray(x, np.float32).reshape(-1, hidden)
    gate, up = np.asarray(w, np.float32).reshape(-1, 2, rows, hidden).swapaxes(0, 1)
    g, u = np.einsum("crk,ck->cr", gate, x), np.einsum("crk,ck->cr", up, x)
    with np.errstate(over="ignore"):
        return g / (1 + np.exp(-g)) * u


def expert_gate_up(*, rows: int = 4, hidden: int = 256) -> ExternalFunction:
    """One tile of an expert's gated activation: silu of each gate row's dot with x, times its up row's.

    ``expert_gate_up(x, w, h)`` takes ``hidden`` bf16 ``x``, the ``rows``
    gate rows of one expert's ``gate_up`` followed by the ``rows`` up rows
    they pair with (``2 * rows`` rows of ``hidden`` bf16), and writes
    ``rows`` float32 ``h``, which ``expert_down`` reads unrounded.
    """
    _positive("expert_gate_up", rows=rows, hidden=hidden)
    return _make_extern(
        "expert_gate_up",
        _kernel_source("moe/expert_gate_up.cc"),
        [
            np.ndarray[(hidden,), np.dtype[bfloat16]],
            np.ndarray[(2 * rows * hidden,), np.dtype[bfloat16]],
            np.ndarray[(rows,), np.dtype[np.float32]],
        ],
        compile_flags=_STRICT_FP
        + [f"-DEXPERT_ROWS={rows}", f"-DEXPERT_HIDDEN={hidden}"],
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, In, Out),
            reference=partial(expert_gate_up_ref, rows=rows, hidden=hidden),
            tolerance=_EXPERT_TOLERANCE,
            acc_dtype=np.float32,
            reduction=hidden,
        ),
    )


def expert_down_ref(h, w, weight, *, rows: int, width: int):
    """Numpy reference for [`expert_down`][iron.kernels.moe.expert_down], from a zeroed ``y``."""
    h = np.asarray(h, np.float32).reshape(-1, width)
    w = np.asarray(w, np.float32).reshape(-1, rows, width)
    scale = np.asarray(weight, np.float32).reshape(-1, 2)[:, :1]
    return scale * np.einsum("crk,ck->cr", w, h)


def expert_down(*, rows: int = 4, width: int = 256) -> ExternalFunction:
    """One tile of an expert's down projection, weighted and accumulated.

    ``expert_down(h, w, weight, y)`` adds ``weight[0] * (w @ h)`` to ``rows``
    float32 ``y``: ``h`` is ``width`` float32 from ``expert_gate_up``, ``w``
    ``rows`` rows of one expert's ``down`` (``width`` bf16 each) and
    ``weight`` the router's padded pair for that expert, as ``router_top_k``
    writes it. Summing every chosen expert into one ``y`` and rounding it
    once (``convert_copy``) gives a row of the experts' output.
    """
    _positive("expert_down", rows=rows, width=width)
    return _make_extern(
        "expert_down",
        _kernel_source("moe/expert_down.cc"),
        [
            np.ndarray[(width,), np.dtype[np.float32]],
            np.ndarray[(rows * width,), np.dtype[bfloat16]],
            np.ndarray[(2,), np.dtype[bfloat16]],
            np.ndarray[(rows,), np.dtype[np.float32]],
        ],
        compile_flags=[f"-DEXPERT_ROWS={rows}", f"-DEXPERT_WIDTH={width}"],
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, In, In, InOut),
            reference=partial(expert_down_ref, rows=rows, width=width),
            initializers=((3, _zero_y),),
            tolerance=_EXPERT_TOLERANCE,
            acc_dtype=np.float32,
            reduction=width,
        ),
    )
