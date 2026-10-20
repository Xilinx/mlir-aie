#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Mirror of kernels/llama_attn_block.cc: block-wise attention over a KV cache.

Each block of up to `tb` rows is attended the way llama_flowkv_mh attends a
whole cache, then folded into a running (max, sum, normalized sv) state.
Blocks split across `n_cores` contiguous ranges; each core folds its blocks
in order and the merge folds the cores' states in core order, so the result
depends on (tb, n_cores) exactly as the device computes it.
"""

import aie_fp
import numpy as np

FLT_MIN = np.float32(np.finfo(np.float32).tiny)
EXP2_TAYLOR = [
    "1.017808628e-07",
    "1.321548666e-06",
    "1.525273365e-05",
    "1.540352969e-04",
    "1.333355787e-03",
    "9.618128650e-03",
    "5.550410971e-02",
    "2.402265072e-01",
    "6.931471825e-01",
    "1.0",
]


def mul_rne(a, b):
    """Mirrors `vec_mul_rne`: the IEEE fp32 product, zero below FLT_MIN.

    Args:
        `a`: fp32 scalar or array.
        `b`: fp32 scalar or array, broadcastable with `a`.

    Returns:
        The fp32 products, elementwise.
    """
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    exact = a.astype(np.float64) * b.astype(np.float64)
    tiny = (np.abs(a) < FLT_MIN) | (np.abs(b) < FLT_MIN) | (np.abs(exact) < FLT_MIN)
    return np.where(tiny, np.copysign(np.float32(0), exact), (a * b)).astype(np.float32)


def sw_exp_nonpos(x):
    """Mirrors the kernel's `sw_exp_nonpos` (e^x for x <= 0).

    Args:
        `x`: fp32 scalar, at most 0.

    Returns:
        The fp32 value the device computes.
    """
    t = np.float32(np.float32(x) * np.float32("1.442695022"))
    if t < np.float32(-125.0):
        return np.float32(0.0)
    k = int(np.trunc(t))
    f = aie_fp.aie_add(t, -np.float32(k))
    p = np.float32(EXP2_TAYLOR[0])
    for c in EXP2_TAYLOR[1:]:
        p = aie_fp.aie_add(np.float32(p * f), np.float32(c))
    return (np.float32(p).view(np.int32) + np.int32(k << 23)).view(np.float32)


def attend_block(q_i8, q_scale, k_i8, v_i8, k_slot, v_slot, lut_exp):
    """One Q head over one block's rows, as `llama_attn_block` computes it.

    Args:
        `q_i8`: int8 (head_dim,).
        `q_scale`: fp32 scale of `q_i8`.
        `k_i8`: int8 (n, head_dim).
        `v_i8`: int8 (n, head_dim).
        `k_slot`: fp32 (n,) per-row K scales.
        `v_slot`: fp32 (n,) per-row V scales.
        `lut_exp`: fp32 (256,) exp LUT.

    Returns:
        (block max, block sum, normalized sv fp32 (head_dim,)).
    """
    dots = (k_i8.astype(np.int32) @ q_i8.astype(np.int32)).astype(np.float32)
    qk = mul_rne(mul_rne(np.float32(q_scale), k_slot), np.float32(0.125))
    scores = mul_rne(dots, qk)
    m = np.float32(scores.max())
    inv_q = np.float32(1.0) / np.float32(0.05)
    q = aie_fp.round_to_i8(mul_rne(aie_fp.aie_add(scores, -m), inv_q))
    e = lut_exp[q.astype(np.int32) + 128].astype(np.float32)
    s = np.float32(0.0)
    for ei in e:
        s = aie_fp.aie_add(s, ei)
    pv = mul_rne(mul_rne(e, aie_fp.sw_recip(s)), v_slot)
    acc = np.zeros(q_i8.shape[0], np.float32)
    for r in range(len(e)):
        acc = aie_fp.aie_add(acc, mul_rne(v_i8[r].astype(np.float32), pv[r]))
    return m, s, acc


def fold(state, m_b, s_b, sv_b):
    """Folds a block (or a core's state) into `state` as the kernel's `fold`.

    Args:
        `state`: None or (max, sum, sv).
        `m_b`: fp32 max of the incoming part.
        `s_b`: fp32 sum of the incoming part.
        `sv_b`: fp32 (head_dim,) normalized sv of the incoming part.

    Returns:
        The new (max, sum, sv).
    """
    if state is None:
        return m_b, s_b, sv_b
    m, s, sv = state
    big = max(m, m_b)
    a = np.float32(s * sw_exp_nonpos(aie_fp.aie_add(m, -big)))
    b = np.float32(s_b * sw_exp_nonpos(aie_fp.aie_add(m_b, -big)))
    den = aie_fp.aie_add(a, b)
    inv = aie_fp.sw_recip(den)
    wa = np.float32(a * inv)
    wb = np.float32(b * inv)
    return big, den, aie_fp.aie_add(mul_rne(sv, wa), mul_rne(sv_b, wb))


def core_ranges(n_blocks, n_cores):
    """The contiguous block range each core attends: [c*n//C, (c+1)*n//C).

    Args:
        `n_blocks`: blocks in the live cache.
        `n_cores`: cores sharing them.

    Returns:
        A (first, end) pair per core.
    """
    return [
        (c * n_blocks // n_cores, (c + 1) * n_blocks // n_cores) for c in range(n_cores)
    ]


def attention(
    q_chunk_i8,
    q_scales,
    k_i8,
    v_i8,
    k_slot,
    v_slot,
    t,
    tb,
    n_cores,
    lut_exp,
    interleaved=False,
):
    """`llama_attn_block` over the cache's blocks, then `llama_attn_merge_selfcal`.

    Args:
        `q_chunk_i8`: int8 (rep, head_dim).
        `q_scales`: fp32 (rep,).
        `k_i8`: int8 (t, head_dim).
        `v_i8`: int8 (t, head_dim).
        `k_slot`: fp32 (t,).
        `v_slot`: fp32 (t,).
        `t`: live rows.
        `tb`: rows per block.
        `n_cores`: cores the blocks are split across.
        `lut_exp`: fp32 (256,) exp LUT.
        `interleaved`: core c takes blocks c, c + n_cores, ... instead of a
            contiguous range (the array decode's split, where every core sees
            every block).

    Returns:
        (int8 (rep, head_dim) output, fp32 (rep,) sv_out_scales, fp32 (rep, head_dim) sv).
    """
    n_blocks = -(-t // tb)
    per_core = (
        [range(c, n_blocks, n_cores) for c in range(n_cores)]
        if interleaved
        else [range(first, end) for first, end in core_ranges(n_blocks, n_cores)]
    )
    rep = q_chunk_i8.shape[0]
    out = np.zeros(q_chunk_i8.shape, np.int8)
    scales = np.zeros(rep, np.float32)
    svs = np.zeros(q_chunk_i8.shape, np.float32)
    for h in range(rep):
        merged = None
        for blocks in per_core:
            state = None
            for b in blocks:
                rows = slice(b * tb, min(t, (b + 1) * tb))
                state = fold(
                    state,
                    *attend_block(
                        q_chunk_i8[h],
                        q_scales[h],
                        k_i8[rows],
                        v_i8[rows],
                        k_slot[rows],
                        v_slot[rows],
                        lut_exp,
                    ),
                )
            if state is not None:
                merged = fold(merged, *state)
        if merged is None:
            raise ValueError("attention needs at least one live row")
        sv = merged[2]
        absmax = np.float32(max(float(np.abs(sv).max()), 1e-12))
        scale = np.float32(absmax * np.float32(1.0 / 127.0))
        out[h] = aie_fp.round_to_i8(mul_rne(sv, aie_fp.sw_recip(scale)))
        scales[h] = scale
        svs[h] = sv
    return out, scales, svs
